
using namespace std;

struct node_t {
    float coords[4];
    int32_t seq_length;
};
struct node_data_t {
    uint32_t node_count;
    vector<node_t> nodes;
};


struct path_element_t {
    uint32_t pidx;
    uint32_t node_id;
    int64_t pos;    // if position negative: reverse orientation
};

struct path_t {
    uint32_t step_count;
    uint64_t first_step_in_path;  // precomputed position in path
    vector<path_element_t> elements;
};

struct path_data_t {
    uint32_t path_count;
    uint32_t total_path_steps;
    vector<path_t> paths;
    vector<path_element_t> element_array;
};

//void cuda_layout(layout_config_t config, const odgi::graph_t &graph, std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y);
