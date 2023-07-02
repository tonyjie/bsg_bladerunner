#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>
#include <sstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include "dirty_zipfian_int_distribution.h"
#include <sys/types.h>
#include <unistd.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

double dist(std::vector<double> &point_1, std::vector<double> &point_2)
{
    return sqrt(pow(point_1[0]-point_2[0],2) + pow(point_1[1] - point_2[1],2));
}

double stress(std::vector<std::vector<std::vector<int>>> &paths, std::vector<std::vector<double>> &layout)
{
    double stress = 0;
    for(auto path : paths)
    {
        for(int i = 0; i < path.size() - 1; i++)
            stress += pow(dist(layout[path[i][0]],layout[path[i+1][0]]) - abs(path[i][1] - path[i+1][1]),2);
    }
    return stress;
}

std::vector<double> path_linear_sgd_layout_schedule(const double &w_min,
                                                    const double &w_max,
                                                    const uint64_t &iter_max,
                                                    const uint64_t &iter_with_max_learning_rate,
                                                    const double &eps) 
{
    double eta_max = 1.0 / w_min;
    double eta_min = eps / w_max;
    double lambda = log(eta_max / eta_min) / ((double) iter_max - 1);
    // initialize step sizes
    std::vector<double> etas;
    etas.reserve(iter_max+1);
    for (int64_t t = 0; t <= iter_max; t++) {
        etas.push_back(eta_max * exp(-lambda * (abs(t - (int64_t) iter_with_max_learning_rate))));
    }
    return etas;
}

void lay(std::vector<std::vector<std::vector<int>>> &paths, std::vector<double> &X, std::vector<double> &Y, int max_iter, int id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    double eta_max = 17000*17000;
    double w_min = (double) 1.0 / (double) (eta_max);
    double w_max = 1.0;
    double iter_max = 6000;
    uint64_t iter_with_max_learning_rate = 0;
    double eps = 0.1;
    std::vector<double> etas = path_linear_sgd_layout_schedule (w_min, w_max, iter_max,
                                                                iter_with_max_learning_rate,
                                                                eps);


    // cache zipf zetas for our full path space
    uint64_t space = 0;
    for(auto path : paths) space = std::max(space, path.size());
    int space_max = 1000;
    int space_quantization_step = 100;
    std::vector<double> zetas((space <= space_max ? space : space_max + (space - space_max) / space_quantization_step + 1)+1);
    double zeta_tmp = 0.0;
    double theta =  0.99;
    for (uint64_t i = 1; i < space + 1; i++) {
        zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / i, theta);
        if (i <= space_max) {
            zetas[i] = zeta_tmp;
        }
        if (i >= space_max && (i - space_max) % space_quantization_step == 0) {
            zetas[space_max + 1 + (i - space_max) / space_quantization_step] = zeta_tmp;
        }
    }
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd() + id*10); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<uint64_t> flip(0, 1);
    for(int eta_idx = 0; eta_idx < etas.size() ; eta_idx++)
    {   
        auto eta = etas[eta_idx];
        for(int iter = 0; iter < max_iter; iter++)
        {    
            auto path = rand()%paths.size();
            int node_a_idx = rand()%paths[path].size();
            //Cooling and Zipfian Stuff
            int node_b_idx;
            if (eta_idx >= 0.9*etas.size() || flip(gen)) 
            {
                if (node_a_idx > 0 && flip(gen) || node_a_idx == paths[path].size()-1) {
                    // go backward
                    uint64_t jump_space = std::min(space, (uint64_t) node_a_idx);
                    uint64_t space = jump_space;
                    if (jump_space > space_max){
                        space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                    }
                    dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, zetas[space]);
                    dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                    uint64_t z_i = z(gen);
                    node_b_idx = node_a_idx - z_i;
                    assert(node_b_idx >= 0);
                } else {
                    // go forward
                    uint64_t jump_space = std::min(space, (uint64_t) (paths[path].size() - node_a_idx - 1));
                    uint64_t space = jump_space;
                    if (jump_space > space_max){
                        space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                    }
                    dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, zetas[space]);
                    dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                    uint64_t z_i = z(gen);
                    //assert(z_i <= path_space);
                    node_b_idx = node_a_idx + z_i;
                    assert(node_b_idx < paths[path].size());
                }
                
            } 
            else 
            {
                node_b_idx = rand()%paths[path].size();
            }
            auto pos_in_path_a = paths[path][node_a_idx][1];
            auto pos_in_path_b = paths[path][node_b_idx][1];
            double term_dist = std::abs(
                static_cast<double>(pos_in_path_a) - static_cast<double>(pos_in_path_b));
            if (term_dist == 0) {
                continue;
            }
            double term_weight = 1.0 / (double) term_dist;
            double w_ij = term_weight;
            double mu = eta * w_ij;
            if (mu > 1) {
                mu = 1;
            }
            double d_ij = term_dist;
            double dx = X[paths[path][node_a_idx][0]] - X[paths[path][node_b_idx][0]];
            double dy = Y[paths[path][node_a_idx][0]] - Y[paths[path][node_b_idx][0]];
            if (dx == 0) {
                dx = 1e-9; // avoid nan
            }
            if(dy == 0) {
                dy = 1e-9;
            }
            double mag = sqrt(dx * dx + dy * dy);
            double Delta = mu * (mag - d_ij) / 2;
            double Delta_abs = std::abs(Delta);
            double r = Delta / mag;
            double r_x = r * dx;
            double r_y = r * dy;
            X[paths[path][node_a_idx][0]] -= r_x;
            Y[paths[path][node_a_idx][0]] -= r_y;
            X[paths[path][node_b_idx][0]] += r_x;
            Y[paths[path][node_b_idx][0]] += r_y;
        }
    }
}

int main(int argc, char* argv[])
{
    /*
        In this simple implementation the input is in the format of a file containing the graph edges.
        The first line describes the input it contains the number of nodes and the number of lines following it.
        The rest of the lines have the format 
            i j d(i,j)
        i is the number of the first node, j is the number of another node, and d(i,j) is the distance between the two nodes.
        the calculation of the distances is performed using dijkstra's algorithm. I decided to take the input with precalcualted
        path lengths since the bottleneck in the layout algorithm is the SGD, the calculation of the shortests paths can also be parallelized
        but that's not the focus rn.
    */

    std::cerr << getpid() << std::endl;
    auto t1 = high_resolution_clock::now();

    std::ifstream graph ("psgd.mtx");
    std::vector<std::vector<std::vector<int>>> paths;
    std::string line;
    int max;
    std::cout << "------Input Summary------\n";
    while (std::getline(graph, line))
    {
        std::istringstream iss(line);
        std::vector<std::vector<int>> path;
        int a, b;
        while ((iss >> a >> b)) { 
            path.push_back({a,b}); 
            max = std::max(a,max);
        }
        paths.push_back(path);
    }
    std::cout<< paths.size() << std::endl;
    for(auto i : paths)
    {
        std::cout << i.size() << std::endl;
    }
    std::cout << max << std::endl;
    std::vector<double> X(max+1);
    std::vector<double> Y(max+1);
    X[0] = 0;
    Y[0] = 0;
    for(int i = 0; i < max; i++)
    {
        X[i] = (rand()/10000.);
        Y[i] = (rand()/10000.);
    }
    std::cout << "------End of Summary------" << std::endl;
    int numThreads = atoi(argv[1]);
    std::thread threads[numThreads];
    for (int i = 0; i < numThreads; ++i) {
        threads[i] = std::thread(lay, std::ref(paths), std::ref(X), std::ref(Y),atoi(argv[2]),i);
    }

    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    for(int i = 0; i < max + 1 ; i++)
    {
        std::cout << std::setprecision(10) << X[i] << " " << Y[i] << '\n';
    }
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cerr << ms_double.count() << "ms\n";
    return 0;
}
