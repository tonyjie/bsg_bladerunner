/*
 * This kernel generates sparse matrix as output rather than dense matrix
 */

// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included. bsg_tiles_X and bsg_tiles_Y must also be defined for
// legacy reasons, but they are deprecated.
#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROUP_Y_DIM 1
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>
#include <bsg_tile_group_barrier.h>
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

/* We wrap all external-facing C++ kernels with `extern "C"` to
 * prevent name mangling
 */
extern "C" {

// A: m by k, CSR
// B: vector
// Out: m size dense vector
int  __attribute__ ((noinline)) kernel_spgemm(sparse_mat_t *A,
                                              sparse_mat_t *B,
                                              sparse_mat_t *Out,
                                              int A_Height,
                                              int A_Width,
                                              int B_Width) {
  // get the absolute tid (which group then which tid within that group)
  int thread_num= bsg_tiles_X * bsg_tiles_Y; 
  int num = A->nnz; 
  int length = int((float) num/(float) thread_num);
  int start = (__bsg_id * length)<num? (__bsg_id * length) : num; 
  int end = (start + length) > num? num : (start+ length);                                  
                                                               
  bsg_cuda_print_stat_kernel_start();                                         
                                        
  //bsg_printf("Start Calculating nnz! \n");
  int nnz =0; 
  int A_row_start = 0; 
  int Upperbounds[thread_num]; 
  for (int i=0; i<A_Height; i++){
    if (sparse_mat_get_idx_ptr(A, i)>start||sparse_mat_get_idx_ptr(A, i)==start){
      A_row_start = i-1; 
      break; 
    }
  }
     
  int A_row_end = 0; 
   for (int i=0; i<A_Height; i++){
     if (sparse_mat_get_idx_ptr(A, i)>end||sparse_mat_get_idx_ptr(A, i)==end){
      A_row_end = i-1; 
      break; 
    }
  }
   
  int zero_rows = 0;
  for (int i=A_row_start; i<=A_row_end; i++){
    if (sparse_mat_get_idx_ptr(A, i)==sparse_mat_get_idx_ptr(A, i+1)){
      zero_rows  = zero_rows+1;
    }
  }
   
  int zero_rows_num = zero_rows *B_Width;
  int zero_cols =0; 
  int *cols[B_Width]; 
  for (int i=0; i<B->nnz; i++){
    cols[sparse_mat_get_col(B, i)]=cols[sparse_mat_get_col(B, i)]+1; 
  }
   
  for(int i=0; i<B_Width; i++){
    if (cols[i]==0){
      zero_cols=zero_cols+1; 
    }
  }

  int total_rows = A_row_end-A_row_start+1; 
  nnz = total_rows*B_Width - zero_rows_num - zero_cols*(total_rows-zero_rows);
  Upperbounds[__bsg_id]=nnz;
  g_barrier.sync(); 


  int before_index; 

  for (int i=0; i<__bsg_id; i++){
    before_index += Upperbounds[i]; 
  }

  for (int row_A_id = A_row_start; row_A_id <= A_row_end; row_A_id ++){
    for (int col_B_id = 0; col_B_id < B_Width; col_B_id ++){
      int sum = 0;
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
      
      if (sum!=0){
        Out->idx_ptrs[row_A_id+1]= Out->idx_ptrs[row_A_id+1] +1;
        int new_index = -1;
        for (int l = A_row_start+1; l < row_A_id+2; l++){
          new_index = new_index + Out->idx_ptrs[l];
        }
        Out->nz_cols[new_index+before_index]= col_B_id;
        Out->nz_data[new_index+before_index]= sum;
      }
    }
  }
  
  g_barrier.sync(); 
  for (int i = 1; i < total_rows; i++) {
    Out->idx_ptrs[i] = Out->idx_ptrs[i] + Out->idx_ptrs[i - 1];
  }
  bsg_tile_group_barrier(&r_barrier, &c_barrier);
  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
 
  return 0;
 }

} // extern "C"
