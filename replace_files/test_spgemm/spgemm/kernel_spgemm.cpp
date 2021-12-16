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
// Out: m by n, CSR
int  __attribute__ ((noinline)) kernel_spgemm(sparse_mat_t *A,
                                              sparse_mat_t *B,
                                              sparse_mat_t *Out,
                                              int A_Height,
                                              int A_Width,
                                              int B_Width,
                                              int *tile_boundary,
                                              int *col_idx_temp,
                                              data_t *val_temp,
                                              int *tile_nnz) {
  int thread_num =bsg_tiles_X * bsg_tiles_Y; 
  int length = ceil((float) A_Height/(float) thread_num);
  int A_start_row = (__bsg_id * length)<A_Height? (__bsg_id * length) : A_Height; 
  int A_end_row = (A_start_row + length) > A_Height? A_Height : (A_start_row + length);   

  data_t sum;
  int curr_nnz = 0;
  bsg_cuda_print_stat_kernel_start();

  //Iterate over rows of A 
  for (int row_A_id = A_start_row; row_A_id < A_end_row; row_A_id++) {
    //Iterate over cols of B 
    for (int col_B_id = 0; col_B_id < B_Width; col_B_id++) {
      
      sum = 0;
      for (int i = sparse_mat_get_idx_ptr(A,row_A_id); i < sparse_mat_get_idx_ptr(A,row_A_id + 1); i++) {
        int col_A_id = sparse_mat_get_col(A,i);

        // get value of B at row=col_A_id and col=col_B_id)
        int k = -1;
        // look at each nonzero value in that row of B
        for (int j = sparse_mat_get_idx_ptr(B, col_A_id); j < sparse_mat_get_idx_ptr(B, col_A_id + 1); j++) {
          if (sparse_mat_get_col(B, j) == col_B_id) {
            // if we find a matching column, store its index
            k = j;
          }
        }
        // only do this computation if a match was found
        if (k != -1) {
          // index of B in CSR form is index for current row + index of col (found above)
          int B_idx = sparse_mat_get_idx_ptr(B, col_A_id) + k;
          sum += sparse_mat_get_val(A, i) * sparse_mat_get_val(B,B_idx);
        }
      }

      // if sum is nonzero, increase curr_nnz by 1, then write result to correct index in temp array temp_val
      if (sum!=0){
        Out->idx_ptrs[row_A_id+1]++; 
        val_temp[tile_boundary[__bsg_id]+curr_nnz]=sum; 
        col_idx_temp[tile_boundary[__bsg_id]+curr_nnz]=col_B_id;
        curr_nnz++; 
      }
    }
  }

  // update tile_nonzero value with curr_nnz so each tile knows how many nnz the others have
  tile_nnz[__bsg_id] = curr_nnz;

  barrier.sync();

  //Update output idx_ptr
  if (__bsg_id == 0)
    for (int i = 1; i < A_Height + 1; i++)
      Out->idx_ptrs[i] += Out->idx_ptrs[i-1];

  barrier.sync();

  // each tile looks at previous entry in tile_nonzero to figure out where its start index in output array is
  
  if (A_start_row < A_end_row){
    int nnz_before =0; 
    for (int j=0; j<__bsg_id; j++){
      nnz_before = tile_nnz[j]+nnz_before; 
    }
    
    for (int i=0; i<curr_nnz; i++){
     Out->nz_cols[nnz_before+ i]=col_idx_temp[tile_boundary[__bsg_id]+ i];
     Out->nz_data[nnz_before+ i]=val_temp[tile_boundary[__bsg_id]+i];
    }
  }
  
  barrier.sync();

  if (__bsg_id == 0){
    Out->m=A_Height; 
    Out->n=B_Width; 
    Out->nnz=tile_boundary[128];
  }
  
  bsg_cuda_print_stat_kernel_end();
  barrier.sync();

  return 0;
}

} // extern "C"
