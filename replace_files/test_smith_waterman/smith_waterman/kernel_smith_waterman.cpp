/*
 * This kernel performs spgemm. 
 * =========================== Feature : =========================
 * Using inner-product method;
 * Dense matrix non-transposed;
 * Distributes Dense matrix columns to tiles;
 * Not implementing any optimization;
 * Could support arbitrary matrix size;
 * Returns output as sparse matrix
*/
  
// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included. bsg_tiles_X and bsg_tiles_Y must also be defined for
// legacy reasons, but they are deprecated.
#define BSG_TILE_GROUP_X_DIM 16
#define BSG_TILE_GROUP_Y_DIM 8
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>
#include <bsg_set_tile_x_y.h>
#include <bsg_tile_group_barrier.hpp>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

bsg_barrier<bsg_tiles_X, bsg_tiles_Y> barrier;

extern int bsg_printf(const char*, ...);

int mm(const char a, const char b, int match_score, int mismatch_score) { 
  return (a==b)?match_score:mismatch_score; 
}

extern "C" {

int  __attribute__ ((noinline)) kernel_smith_waterman(char *seqa,
                                              char *seqb,
                                              int *matrix,
                                              int m,
                                              int n,
                                              int match_score,
                                              int mismatch_score,
                                              int gap) {
  int thread_num =bsg_tiles_X * bsg_tiles_Y; 
  bsg_cuda_print_stat_kernel_start();
 
  if(__bsg_id == 0) {
    for(int i = 1; i < m+1; i++) {
      for(int j = 1; j < n+1; j++) {
        int up = matrix[(i-1)*(n+1)+j] + gap;
        int left = matrix[i*(n+1)+j-1] + gap;
        int temp = std::max(up, left);
        temp = std::max(temp, 0);
        int value = mm(seqb[i-1], seqa[j-1], match_score, mismatch_score);
        int diag = matrix[(i-1)*(n+1)+(j-1)] + value;
        matrix[i*(n+1)+j] = std::max(temp, diag);
      }
    }
  } 
  bsg_cuda_print_stat_kernel_end();
  barrier.sync();

  return 0;
}

} // extern "C"
