

/*
 * This kernel performs spgemm. 
 * =========================== Feature : =========================
 * Using inner-product method;
 * One of the sparse matrix is transposed;
 * Implement one relative simple optimization;
 * Could support arbitrary matrix size;
 */

// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included. bsg_tiles_X and bsg_tiles_Y must also be defined for
// legacy reasons, but they are deprecated.
#define BSG_TILE_GROUP_X_DIM 4
#define BSG_TILE_GROUP_Y_DIM 4
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>
#include <bsg_tile_group_barrier.h>
#include <cstdint>
#include <cstring>
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);

/////////////////////////////////////////////////
// CSR sparse matrix data structure and utilities
/////////////////////////////////////////////////
typedef float data_t;

typedef struct sparse_mat_t {
  // number of non-zero
  int nnz;

  // the dimensions of the matrix
  // m is x, n is y
  int m;
  int n;

  // the non-zero values in the matrix
  data_t *nz_data;

  // y(n) + 1 size, the number of non-zero elements in the ALL of previous rows
  // this is used for lookup into into a row
  int *idx_ptrs;

  // nnz size, the column of the correspnding non-zero value
  int *nz_cols;

} __attribute__((packed)) sparse_mat_t;

void sparse_mat_set_nz_val(sparse_mat_t *mat, int idx, data_t val) {
  (mat->nz_data)[idx] = val;
}

int sparse_mat_get_col(sparse_mat_t *mat, int idx) {
  return mat->nz_cols[idx];
}

int sparse_mat_get_idx_ptr(sparse_mat_t *mat, int row) {
  return mat->idx_ptrs[row];
}

data_t sparse_mat_get_val(sparse_mat_t *mat, int idx) {
  return mat->nz_data[idx];
}

extern int bsg_printf(const char*, ...);

extern "C" {

// A: m by n, CSR
// B: n by k, CSR
// Out: m by n, dense
int  __attribute__ ((noinline)) kernel_spgemm(sparse_mat_t *A,
                                              sparse_mat_t *B,
                                              data_t *Out,
                                              int A_Height,
                                              int A_Width,
                                              int B_Width) {
  // get the absolute tid (which group then which tid within that group)
  int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
  int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
  int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x;
  int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y;

  int num_threads = num_threads_x * num_threads_y;
  int B_Height = A_Width;
  int tid = tid_y * num_threads_x + tid_x;

  data_t sum;
 
  bsg_cuda_print_stat_kernel_start();

  // new version: sparse matrix * transpose sparse matrix
  for (int row_B_id = tid; row_B_id < A_Width; row_B_id += num_threads){
    for (int row_A_id = 0; row_A_id < A_Height; row_A_id ++){
      sum = 0;
      for (int i = sparse_mat_get_idx_ptr(A,row_A_id); i < sparse_mat_get_idx_ptr(A,row_A_id + 1); i++) {
        int col_A_id = sparse_mat_get_col(A,i);
        // get value of B at col=col_A_id and row=row_B_id)
        int k = -1;
        // look at each nonzero value in that row of B
        for (int j = sparse_mat_get_idx_ptr(B, row_B_id); j < sparse_mat_get_idx_ptr(B, row_B_id + 1); j++) {
          if (sparse_mat_get_col(B, j) == col_A_id) {
            // if we find a matching column, store its index
            k = j;
          }
        } 
        // only do this computation if a match was found
        if (k != -1) {
          // index of B in CSR form is index for current row + index of col (found above)
          int B_idx = sparse_mat_get_idx_ptr(B, row_B_id) + k;
          sum += sparse_mat_get_val(A, i) * sparse_mat_get_val(B,B_idx);
        }
      }
      Out[row_A_id * A_Width + row_B_id] = sum;
    }
  }

  bsg_tile_group_barrier(&r_barrier, &c_barrier);
  bsg_cuda_print_stat_kernel_end();
  
  return 0;
}

} // extern "C"

