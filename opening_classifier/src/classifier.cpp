#include "classifier.hpp"
#include "movegen.hpp"
#include <queue>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <atomic>
#include <unordered_set>

using namespace std;

void ClassifierEngine::load_eco(const vector<tuple<string,string,string>>& rows){

    roots_.clear();

    for (auto& [eco, name, fen] : rows) {
        try {
            Board b = board_from_fen(fen);
            roots_.push_back({eco, name, b, default_prior_});
            auto& last = roots_.back();
            if (last.board.occupied == 0 && last.board.zobrist != 0)
                cerr << "CORRUPT board for " << eco << "\n";
        } catch (...) {
            cerr << "Skipping invalid fen: " << fen << "\n";
        }
    }
    cout << "Loaded " << roots_.size() << " ECO roots.\n";
}

void ClassifierEngine::load_priors(const unordered_map<string, double>& priors)
{
    priors_ = priors;
    cout << "Loading default priors with default prior:" << default_prior_ << endl;
    double min_p = 1.0;
    for (auto& [eco, p] : priors)
        if (p > 0 && p < min_p) min_p = p;
    floor_prior_ = min_p * 0.1;

    for (auto& root : roots_) {
        auto it = priors_.find(root.eco);
        root.prior = (it != priors_.end()) ? it->second : default_prior_;
    }
}

void ClassifierEngine::load_book(const vector<const char*> paths){
    book.Clear();
    book.Load(paths);
}

void ClassifierEngine::build_index(int max_depth, double min_log_prob) {
    reach_index_.clear();
    board_zh_.clear();

    Board start = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    generate_legal_scored_moves(start, 0, book);

    int total = (int)roots_.size();
    int nthreads = omp_get_max_threads();

    cout << "Build Index BFS initiated with " << total << " ECOs" << " across " << nthreads << " threads\n";

    vector<unordered_map<uint64_t, vector<ReachEntry>>> local_indices(nthreads);
    vector<vector<Board>> local_board_zh(nthreads);

    atomic<int> done{0};
    auto wall_start = chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < total; i++) {
        const auto& root = roots_[i];
        struct Node { 
            Board board; 
            double log_prob; 
            int depth; 
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
            local_board_zh[omp_get_thread_num()].push_back(board);

            if (depth >= max_depth) continue;

            auto scored_moves = generate_legal_scored_moves(board, depth, book);
            int  n = (int)scored_moves.size();
            if  (n == 0) continue;

            
            for (auto& scored_move : scored_moves) {
                Move mv = scored_move.first;
                double score = scored_move.second;
                Board child = apply_move(board, mv);
                uint64_t czh = child.zobrist;
                
                auto vit = visited.find(czh);
                // add check to verify if it is not another Opening ECO

                double child_lp = lp + log(score);
                if (child_lp < min_log_prob) continue;

                prob_acc[czh] += exp(child_lp);
                
                if (!shortest.count(czh))
                    shortest[czh] = depth + 1;

                if (vit == visited.end() || vit->second < child_lp) {
                    visited[czh] = child_lp;
                    q.push({child, child_lp, depth + 1});
                }
            }
        }

        int tid = omp_get_thread_num();
        for (auto& [zh, prob] : prob_acc) {
            int pl = shortest.count(zh) ? shortest.at(zh) : -1;
            local_indices[tid][zh].push_back({ root.eco, (float)prob, (uint8_t)(pl < 0 ? 255 : pl) });
        }

        int n_done = ++done;
        #pragma omp critical
        {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - wall_start).count();
            double rate = n_done / elapsed;
            double remaining = (total - n_done) / rate;
            cout << "\rProgress: " << n_done << "/" << total
                 << " | Elapsed: " << int(elapsed) << " | ETA: " << int(remaining) << "s   " << flush;
        }
    }

    cout << "\nMerging thread results...\n";
    for (auto& local : local_indices) {
        for (auto& [zh, entries] : local) {
            auto& dest = reach_index_[zh];
            dest.insert(dest.end(),
                        make_move_iterator(entries.begin()),
                        make_move_iterator(entries.end()));
        }
        local.clear();  
    }
    local_indices.clear();  

    unordered_set<uint64_t> seen_zh;
    for (auto& local : local_board_zh){
        for (auto& entry : local){
            if (seen_zh.insert(entry.zobrist).second){
                board_zh_.push_back(entry);
            }
        }
    }    
    cout << "Index built. " << reach_index_.size() << " unique positions indexed.\n";
}

vector<ScoredOpening> ClassifierEngine::classify( const string& fen, int top_n) const{

    Board query = board_from_fen(fen);
    uint64_t qhash = query.zobrist;

    auto it = reach_index_.find(qhash);

    unordered_map<string, double> probabilities;
    unordered_map<string, int> path_lengths;

    for (auto& root : roots_)
        probabilities[root.eco] = EPSILON;

    if (it != reach_index_.end()) {
        for (auto& entry : it->second) {
            probabilities[entry.eco]  = entry.prob;
            path_lengths[entry.eco] = (entry.path_length == 255) ? -1 : entry.path_length;
        }
    }

    double normaliser = 0.0;
    vector<ScoredOpening> results;
    results.reserve(5000);

    for (auto& root : roots_) {
        string eco = root.eco;
        double L = probabilities[eco];

        double prior = floor_prior_;
        auto pit = priors_.find(eco);
        if (pit != priors_.end()) prior = pit->second;

        double unnorm = prior * L;
        normaliser += unnorm;

        results.push_back({
            eco,
            root.name,
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

    uint64_t nroots = roots_.size();
    f.write(reinterpret_cast<const char*>(&nroots), sizeof(nroots));
    for (auto& r : roots_) {
        uint32_t elen = r.eco.size(), nlen = r.name.size();
        f.write(reinterpret_cast<const char*>(&elen), sizeof(elen));
        f.write(r.eco.data(), elen);
        f.write(reinterpret_cast<const char*>(&nlen), sizeof(nlen));
        f.write(r.name.data(), nlen);
        f.write(reinterpret_cast<const char*>(&r.board), sizeof(Board));
        f.write(reinterpret_cast<const char*>(&r.prior), sizeof(r.prior));
    }

    uint64_t n = reach_index_.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (auto& [zh, entries] : reach_index_) {
        f.write(reinterpret_cast<const char*>(&zh), sizeof(zh));
        uint32_t ne = (uint32_t)entries.size();
        f.write(reinterpret_cast<const char*>(&ne), sizeof(ne));
        for (auto& e : entries) {
            uint32_t elen = e.eco.size();
            f.write(reinterpret_cast<const char*>(&elen), sizeof(elen));
            f.write(e.eco.data(), elen);
            f.write(reinterpret_cast<const char*>(&e.prob), sizeof(e.prob));
            f.write(reinterpret_cast<const char*>(&e.path_length), sizeof(e.path_length));
        }
    }

    uint64_t nboard = board_zh_.size();
    f.write(reinterpret_cast<const char*>(&nboard), sizeof(nboard));
    for (auto& b : board_zh_)
        f.write(reinterpret_cast<const char*>(&b), sizeof(Board));

    cout << "Index saved to " << path
         << " (" << nroots << " roots, " << n << " positions, " << nboard << " boards)\n";
}

void ClassifierEngine::load_index(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("Cannot open index file: " + path);

    roots_.clear();
    reach_index_.clear();
    board_zh_.clear();

    uint64_t nroots;
    f.read(reinterpret_cast<char*>(&nroots), sizeof(nroots));
    roots_.resize(nroots);
    for (uint64_t i = 0; i < nroots; i++) {
        uint32_t elen, nlen;
        f.read(reinterpret_cast<char*>(&elen), sizeof(elen));
        roots_[i].eco.resize(elen);
        f.read(roots_[i].eco.data(), elen);
        f.read(reinterpret_cast<char*>(&nlen), sizeof(nlen));
        roots_[i].name.resize(nlen);
        f.read(roots_[i].name.data(), nlen);
        f.read(reinterpret_cast<char*>(&roots_[i].board), sizeof(Board));
        f.read(reinterpret_cast<char*>(&roots_[i].prior), sizeof(roots_[i].prior));
    }

    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    for (uint64_t i = 0; i < n; i++) {
        uint64_t zh;
        f.read(reinterpret_cast<char*>(&zh), sizeof(zh));
        uint32_t ne;
        f.read(reinterpret_cast<char*>(&ne), sizeof(ne));
        auto& vec = reach_index_[zh];
        vec.resize(ne);
        for (uint32_t j = 0; j < ne; j++) {
            uint32_t elen;
            f.read(reinterpret_cast<char*>(&elen), sizeof(elen));
            vec[j].eco.resize(elen);
            f.read(vec[j].eco.data(), elen);
            f.read(reinterpret_cast<char*>(&vec[j].prob), sizeof(vec[j].prob));
            f.read(reinterpret_cast<char*>(&vec[j].path_length), sizeof(vec[j].path_length));
        }
    }

    uint64_t nboard;
    f.read(reinterpret_cast<char*>(&nboard), sizeof(nboard));
    board_zh_.resize(nboard);
    for (uint64_t i = 0; i < nboard; i++)
        f.read(reinterpret_cast<char*>(&board_zh_[i]), sizeof(Board));

    cout << "Index loaded from " << path
         << " (" << nroots << " roots, " << n << " positions, " << nboard << " boards)\n";
}