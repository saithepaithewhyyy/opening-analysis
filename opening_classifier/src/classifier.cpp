#include "classifier.hpp"
#include "movegen.hpp"
#include <queue>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iostream>

using namespace std;

void ClassifierEngine::load_eco(const vector<tuple<string,string,string>>& rows){

    roots_.clear();
    eco_name_.clear();
    all_eco_codes_.clear();

    for (auto& [eco, name, epd] : rows) {
        try {
            Board b = board_from_fen(epd);
            roots_.push_back({eco, name, b, default_prior_});
            eco_name_[eco] = name;
            all_eco_codes_.push_back(eco);
        } catch (...) {
            cerr << "Skipping invalid EPD: " << epd << "\n";
        }
    }
    cout << "Loaded " << roots_.size() << " ECO roots.\n";
}


void ClassifierEngine::load_priors(const unordered_map<string, double>& priors)
{
    priors_ = priors;

    double min_p = 1.0;
    for (auto& [eco, p] : priors)
        if (p > 0 && p < min_p) min_p = p;
    default_prior_ = min_p * 0.1;

    for (auto& root : roots_) {
        auto it = priors_.find(root.eco);
        root.prior = (it != priors_.end()) ? it->second : default_prior_;
    }
}


void ClassifierEngine::build_index(int max_depth, double min_log_prob) {
    reach_index_.clear();

    int total = (int)roots_.size();
    int done  = 0;

    for (auto& root : roots_) {
        // BFS  
        struct Node {
            Board  board;
            double log_prob;
            int    depth;
        };

        unordered_map<uint64_t, double> visited;
        visited[root.board.zobrist] = 0.0;

        queue<Node> q;
        q.push({root.board, 0.0, 0});

        unordered_map<uint64_t, double> prob_acc;
        unordered_map<uint64_t, int> shortest;

        prob_acc[root.board.zobrist] = 1.0;
        shortest[root.board.zobrist] = 0;

        while (!q.empty()) {
            auto [board, lp, depth] = q.front();
            q.pop();

            if (depth >= max_depth) continue;

            auto moves = generate_legal_moves(board);
            int  n = (int)moves.size();
            if  (n == 0) continue;

            double child_lp = lp - log((double)n);
            if (child_lp < min_log_prob) continue;

            for (auto& mv : moves) {
                Board child = apply_move(board, mv);
                uint64_t czh = child.zobrist;

                prob_acc[czh] += exp(child_lp);

                if (!shortest.count(czh))
                    shortest[czh] = depth + 1;

                auto vit = visited.find(czh);
                if (vit == visited.end() || vit->second < child_lp) {
                    visited[czh] = child_lp;
                    q.push({child, child_lp, depth + 1});
                }
            }
        }

        for (auto& [zh, prob] : prob_acc) {
            reach_index_[zh].push_back({
                root.eco,
                prob,
                shortest.count(zh) ? shortest.at(zh) : -1
            });
        }

        done++;
        if (done % 100 == 0)
            cout << "  Indexed " << done << "/" << total << " roots...\n";
    }

    cout << "Index built. " << reach_index_.size() << " unique positions indexed.\n";
}


vector<ScoredOpening> ClassifierEngine::classify( const string& fen, int top_n) const{

    Board query = board_from_fen(fen);
    uint64_t qhash = query.zobrist;

    auto it = reach_index_.find(qhash);

    unordered_map<string, double> likelihoods;
    unordered_map<string, int> path_lengths;

    for (auto& eco : all_eco_codes_)
        likelihoods[eco] = EPSILON;

    if (it != reach_index_.end()) {
        for (auto& entry : it->second) {
            likelihoods[entry.eco]  = entry.likelihood;
            path_lengths[entry.eco] = entry.path_length;
        }
    }

    double normaliser = 0.0;
    vector<ScoredOpening> results;
    results.reserve(all_eco_codes_.size());

    for (auto& eco : all_eco_codes_) {
        double L = likelihoods[eco];

        double prior = default_prior_;
        auto pit = priors_.find(eco);
        if (pit != priors_.end()) prior = pit->second;

        double unnorm = prior * L;
        normaliser   += unnorm;

        results.push_back({
            eco,
            eco_name_.count(eco) ? eco_name_.at(eco) : eco,
            L,
            unnorm,
            path_lengths.count(eco) ? path_lengths.at(eco) : -1
        });
    }

    for (auto& r : results)
        r.posterior = (normaliser > 0) ? r.posterior / normaliser : 0.0;

    partial_sort(
        results.begin(),
        results.begin() + min(top_n, (int)results.size()),
        results.end(),
        [](const ScoredOpening& a, const ScoredOpening& b) {
            return a.posterior > b.posterior;
        });
    results.resize(min(top_n, (int)results.size()));
    return results;
}


void ClassifierEngine::save_index(const string& path) const {
    ofstream f(path, ios::binary);
    if (!f) throw runtime_error("Cannot open index file for writing: " + path);

    uint64_t n = reach_index_.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (auto& [zh, entries] : reach_index_) {
        f.write(reinterpret_cast<const char*>(&zh), sizeof(zh));
        uint32_t ne = entries.size();
        f.write(reinterpret_cast<const char*>(&ne), sizeof(ne));
        for (auto& e : entries) {
            uint32_t elen = e.eco.size();
            f.write(reinterpret_cast<const char*>(&elen), sizeof(elen));
            f.write(e.eco.data(), elen);
            f.write(reinterpret_cast<const char*>(&e.likelihood),  sizeof(e.likelihood));
            f.write(reinterpret_cast<const char*>(&e.path_length), sizeof(e.path_length));
        }
    }
    cout << "Index saved to " << path << "\n";
}

void ClassifierEngine::load_index(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("Cannot open index file: " + path);

    reach_index_.clear();
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));

    for (uint64_t i = 0; i < n; i++) {
        uint64_t zh;
        f.read(reinterpret_cast<char*>(&zh), sizeof(zh));
        uint32_t ne;
        f.read(reinterpret_cast<char*>(&ne), sizeof(ne));
        for (uint32_t j = 0; j < ne; j++) {
            uint32_t elen;
            f.read(reinterpret_cast<char*>(&elen), sizeof(elen));
            string eco(elen, '\0');
            f.read(eco.data(), elen);
            double  likelihood;  int32_t path_length;
            f.read(reinterpret_cast<char*>(&likelihood),  sizeof(likelihood));
            f.read(reinterpret_cast<char*>(&path_length), sizeof(path_length));
            reach_index_[zh].push_back({eco, likelihood, path_length});
        }
    }
    cout << "Index loaded from " << path
              << " (" << reach_index_.size() << " positions)\n";
}