//This kernel multiplies 2 vectors and stores the result in the thrid

#include "dirty_zipfian_int_distribution.h"
#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"


#include "bsg_tile_group_barrier.hpp"

bsg_barrier<bsg_tiles_X, bsg_tiles_Y> barrier;

typedef struct {
    int** paths;
    int* path_idx;
    int path_cnt;
} graph_t;

double mysqrt(double number)
{
  const double ACCURACY=0.000001;
  double lower, upper, guess;
  if (number < 1)
  {
    lower = number;
    upper = 1;
  }
  else
  {
    lower = 1.;
    upper = number;
  }
  while ((upper-lower) > ACCURACY)
  {
    guess = (lower + upper)/2;
    if(guess*guess > number)
    upper =guess;
    else
    lower = guess; 
  }
  return (lower + upper)/2;
}

double zetaf(unsigned long __n, double __theta)
{
    double ans = 0.0;
    for(unsigned long i=1; i<=__n; ++i)
    ans += pow(1.0/i, __theta);
    return ans;
}

uint64_t get_zipf(double a, double b, double theta, double zeta, std::mt19937 &__urng)
{
    double alpha = 1 / (1 - theta);
    double eta = (1 - pow(2.0 / (b - a + 1), 1 - theta)) / (1 - zetaf(2,theta) / zeta);
      
    double u = __urng()/(__urng.max()  +  1.0);
      
    double uz = u * zeta;
    if(uz < 1.0) return a;
    if(uz < 1.0 + pow(0.5, theta)) return a + 1;

    return a + ((b - a + 1) * pow(eta*u-eta+1, alpha));    
}

extern "C" __attribute__ ((noinline))
int kernel_float_vec_mul(int path_cnt, int* path_idx, int** paths, float* X, float* Y, int iter_max, double* etas, double* zetas, int N)//, float* Y, int eta_cnt, double* etas, double* zetas) 
{
	// a seed source for the random number engine

    std::mt19937 gen(23462346 + __bsg_tile_group_id_y * __bsg_grid_dim_x + __bsg_tile_group_id_x);
	int space = 0;
	for(int i = 0; i < path_cnt ; i++) space = std::max(space, path_idx[i]);
    int space_max = 1000;
    int space_quantization_step = 100;
    double theta = 0.99;
    for(int eta_idx = 0; eta_idx < iter_max ; eta_idx++)
    {   
        auto eta = etas[eta_idx];
        for(int iter = 0; iter < 1; iter++)  // CHANGE THIS TO CHANGE THE NUMBER OF INNER LOOP RUNS
        {    
            int path = gen()%path_cnt;
            int node_a_idx = gen()%path_idx[path];
            int node_b_idx;
            //Cooling and Zipfian Stuff
            if (eta_idx >= 0.9 * iter_max || gen()%2) 
            {
                if (node_a_idx > 0 && gen()%2 || node_a_idx == path_idx[path] - 1) {
                    // go backward
                    uint64_t jump_space = std::min(space,  node_a_idx);
                    uint64_t space = jump_space;
                    if (jump_space > space_max){
                        space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                    }
                    uint64_t z_i = get_zipf(1, jump_space, theta, zetas[space],gen);
                    node_b_idx = node_a_idx - z_i;
                } 
                else {
                    // go forward
                    uint64_t jump_space = std::min(space,  (path_idx[path] - node_a_idx - 1));
                    uint64_t space = jump_space;
                    if (jump_space > space_max){
                        space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                    }
                    uint64_t z_i = get_zipf(1, jump_space, theta, zetas[space],gen);
                    node_b_idx = node_a_idx + z_i;
                }
            } 
            else 
            {
                node_b_idx = gen()%path_idx[path];
            }
            int pos_in_path_a = paths[path][node_a_idx*2+1];
            int pos_in_path_b = paths[path][node_b_idx*2+1];
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
            double dx = X[paths[path][node_a_idx*2]] - X[paths[path][node_b_idx*2]];
            double dy = Y[paths[path][node_a_idx*2]] - Y[paths[path][node_b_idx*2]];
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
            X[paths[path][node_a_idx*2]] -= r_x;
            Y[paths[path][node_a_idx*2]] -= r_y;
            X[paths[path][node_b_idx*2]] += r_x;
            Y[paths[path][node_b_idx*2]] += r_y;
        }
    }
	barrier.sync();
  return 0;
}
