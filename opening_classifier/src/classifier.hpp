#pragma once
#include "board.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

using namespace std;

struct ScoredOpening {
    string eco;
    string name;
    double likelihood;
    double posterior;
    int path_length;
};

struct ReachEntry {
    uint16_t eco_idx;
    float likelihood;  
    uint8_t path_length;
};

class ClassifierEngine {
public:
    static constexpr double EPSILON = 1e-15;
    static constexpr double MIN_LOG_PROB = -15.0;
    static constexpr int MAX_DEPTH = 3;

    void load_eco(const vector<tuple<string,string,string>>& rows);
    void load_priors(const unordered_map<string, double>& priors = {});

    void build_index(int max_depth = MAX_DEPTH, double min_log_prob = MIN_LOG_PROB);

    vector<ScoredOpening> classify(const string& fen, int top_n = 5) const;

    void save_index(const string& path) const;
    void load_index(const string& path);

private:
    struct EcoRoot {
        string eco;
        string name;
        Board board;
        double prior;
    };

    vector<EcoRoot> roots_;
    unordered_map<string, double> priors_;
    double default_prior_ = 0.00027100271;
    double floor_prior_ = 0.00027100271;

    unordered_map<uint64_t, vector<ReachEntry>> reach_index_;
    vector<Board> board_zh_;

    unordered_map<string, string> eco_name_;
    vector<string> all_eco_codes_;
    unordered_map<string, uint16_t> eco_to_idx_;
};