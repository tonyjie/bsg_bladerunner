#ifndef __SPARSE_MULTIPLY_HPP_ALL_FUNCTION
#define __SPARSE_MULTIPLY_HPP_ALL_FUNCTION
#include <cstdint>
#include <cstring>
#include <bsg_manycore.h>
#include <bsg_tile_group_barrier.h>

//////////////////////////// sub_functions of the kernel functions start /////////////////////
inline void unrolling_load(
  float *dest,
  float *src,
  int   start_addr
){
  dest[0]  = src[start_addr + 0];
  dest[1]  = src[start_addr + 1];
  dest[2]  = src[start_addr + 2];
  dest[3]  = src[start_addr + 3];
  dest[4]  = src[start_addr + 4];
  dest[5]  = src[start_addr + 5];
  dest[6]  = src[start_addr + 6];
  dest[7]  = src[start_addr + 7];
  dest[8]  = src[start_addr + 8];
  dest[9]  = src[start_addr + 9];
  dest[10] = src[start_addr + 10];
  dest[11] = src[start_addr + 11];
  dest[12] = src[start_addr + 12];
  dest[13] = src[start_addr + 13];
  dest[14] = src[start_addr + 14];
  dest[15] = src[start_addr + 15];

}

inline void compute_unrolling(
  float *dest,
  float elem_A,
  float *B_row
){
  dest[0]  += elem_A * B_row[0];
  dest[1]  += elem_A * B_row[1];
  dest[2]  += elem_A * B_row[2];
  dest[3]  += elem_A * B_row[3];
  dest[4]  += elem_A * B_row[4];
  dest[5]  += elem_A * B_row[5];
  dest[6]  += elem_A * B_row[6];
  dest[7]  += elem_A * B_row[7];
  dest[8]  += elem_A * B_row[8];
  dest[9]  += elem_A * B_row[9];
  dest[10] += elem_A * B_row[10];
  dest[11] += elem_A * B_row[11];
  dest[12] += elem_A * B_row[12];
  dest[13] += elem_A * B_row[13];
  dest[14] += elem_A * B_row[14];
  dest[15] += elem_A * B_row[15];
}

inline void unrolling_store(
  float *dest,
  float *src,
  int   start_addr
){
  dest[start_addr + 0]  = src[0];
  dest[start_addr + 1]  = src[1];
  dest[start_addr + 2]  = src[2];
  dest[start_addr + 3]  = src[3];
  dest[start_addr + 4]  = src[4];
  dest[start_addr + 5]  = src[5];
  dest[start_addr + 6]  = src[6];
  dest[start_addr + 7]  = src[7];
  dest[start_addr + 8]  = src[8];
  dest[start_addr + 9]  = src[9];
  dest[start_addr + 10] = src[10];
  dest[start_addr + 11] = src[11];
  dest[start_addr + 12] = src[12];
  dest[start_addr + 13] = src[13];
  dest[start_addr + 14] = src[14];
  dest[start_addr + 15] = src[15];

}

inline void compute_unrolling_v8(
  float *dest,
  float elem_A,
  float *B_row
){
  dest[0]  += elem_A * B_row[0];
  dest[1]  += elem_A * B_row[1];
}

inline void unrolling_store_to_SM_v10(
  float *dest,
  float *src,
  int   src_start_addr,
  int   dest_start_addr
){
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 0), src[src_start_addr + 0]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 1), src[src_start_addr + 1]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 2), src[src_start_addr + 2]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 3), src[src_start_addr + 3]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 4), src[src_start_addr + 4]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 5), src[src_start_addr + 5]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 6), src[src_start_addr + 6]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 7), src[src_start_addr + 7]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 8), src[src_start_addr + 8]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 9), src[src_start_addr + 9]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 10), src[src_start_addr + 10]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 11), src[src_start_addr + 11]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 12), src[src_start_addr + 12]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 13), src[src_start_addr + 13]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 14), src[src_start_addr + 14]);
  bsg_tile_group_shared_store (float, dest, (dest_start_addr + 15), src[src_start_addr + 15]);

}

inline void unrolling_load_from_SM_v10(
  float *dest,
  float *src,
  int   start_addr
){
  bsg_tile_group_shared_load (float, src, (start_addr + 0), dest[0]);
  bsg_tile_group_shared_load (float, src, (start_addr + 1), dest[1]);
  bsg_tile_group_shared_load (float, src, (start_addr + 2), dest[2]);
  bsg_tile_group_shared_load (float, src, (start_addr + 3), dest[3]);
  bsg_tile_group_shared_load (float, src, (start_addr + 4), dest[4]);
  bsg_tile_group_shared_load (float, src, (start_addr + 5), dest[5]);
  bsg_tile_group_shared_load (float, src, (start_addr + 6), dest[6]);
  bsg_tile_group_shared_load (float, src, (start_addr + 7), dest[7]);
  bsg_tile_group_shared_load (float, src, (start_addr + 8), dest[8]);
  bsg_tile_group_shared_load (float, src, (start_addr + 9), dest[9]);
  bsg_tile_group_shared_load (float, src, (start_addr + 10), dest[10]);
  bsg_tile_group_shared_load (float, src, (start_addr + 11), dest[11]);
  bsg_tile_group_shared_load (float, src, (start_addr + 12), dest[12]);
  bsg_tile_group_shared_load (float, src, (start_addr + 13), dest[13]);
  bsg_tile_group_shared_load (float, src, (start_addr + 14), dest[14]);
  bsg_tile_group_shared_load (float, src, (start_addr + 15), dest[15]);

}

//////////////////////////// sub_functions of the kernel functions end /////////////////////



//////////////////////////// kernel function for v5 start /////////////////////////////
template <typename TA, typename TB>
int  __attribute__ ((noinline)) sparse_multiply_inner_product_small_v5(
      TA *A, TB *B, TB *Out, uint32_t A_HEIGHT, uint32_t A_WIDTH, uint32_t B_WIDTH)
{


  int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
  int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
  int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x;
  int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y;

  int num_threads = num_threads_x * num_threads_y;

  int tid = tid_y * num_threads_x + tid_x;

  TB sum;
  TB data_B_row[A_WIDTH];

  for (int row_B_id = tid; row_B_id < B_WIDTH; row_B_id += num_threads){
    memcpy(data_B_row, &B[row_B_id * A_WIDTH], sizeof(TB)*A_WIDTH); 
    for (int row_A_id = 0; row_A_id < A_HEIGHT; row_A_id ++){
      sum = 0;
      for (int i = sparse_mat_get_idx_ptr(A,row_A_id); i < sparse_mat_get_idx_ptr(A,row_A_id + 1); i++) {
        int col_A_id = sparse_mat_get_col(A,i);
        sum += sparse_mat_get_val(A, i) * data_B_row[col_A_id];
      }
      Out[row_A_id * B_WIDTH + row_B_id] = sum;
    }
  }


 return 0;
}

template <typename TA, typename TB>
int  __attribute__ ((noinline)) sparse_multiply_inner_product_large_v5(
      TA *A, TB *B, TB *Out, uint32_t A_HEIGHT, uint32_t A_WIDTH, uint32_t B_WIDTH)
{


  int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
  int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
  int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x;
  int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y;

  int num_threads = num_threads_x * num_threads_y;

  int tid = tid_y * num_threads_x + tid_x;
  
  int B_HEIGHT = A_WIDTH;
  TB sum;

  int mid_size = 0;

  int block_col_size = 500;
  TB B_local[block_col_size];

  float elem_A;

  // calculate the number of row and col blocks in matrix A and matrix B
  int A_num_blk_per_row = (A_WIDTH + block_col_size - 1) / block_col_size;

  // calculate the dimension of the last block size in each matrix
  int A_last_blk_col = (A_WIDTH % block_col_size == 0) ? block_col_size : (A_WIDTH % block_col_size);

  int tmp_start = 0;
  int tmp_end = 0;
  int flag;

  for (int row_B_id = tid; row_B_id < B_WIDTH; row_B_id += num_threads){
    for (int row_A_id = 0; row_A_id < A_HEIGHT; row_A_id ++){
      int ele_id_row_start = sparse_mat_get_idx_ptr(A,row_A_id);
      int ele_id_row_end   = sparse_mat_get_idx_ptr(A,row_A_id + 1);
      sum = Out[row_A_id * B_WIDTH + row_B_id];
      tmp_start = ele_id_row_start;
      for(int inside_blk_id = 0; inside_blk_id < A_num_blk_per_row; inside_blk_id ++){
        mid_size = (inside_blk_id == A_num_blk_per_row - 1) ? A_last_blk_col : block_col_size;

        memcpy(&B_local, &B[row_B_id * B_HEIGHT + inside_blk_id * block_col_size], sizeof(float) * mid_size);

        int bottom = inside_blk_id * block_col_size;
        int top = inside_blk_id * block_col_size + mid_size;

        for(int k = ele_id_row_start; k < ele_id_row_end; k++){
          int col_A_id = sparse_mat_get_col(A,k);
          if((col_A_id < top) && (col_A_id >= bottom)){
            sum += sparse_mat_get_val(A, k) * B_local[col_A_id - inside_blk_id * block_col_size];
          }
        }

        tmp_start = tmp_end;
      
      }
      Out[row_A_id * B_WIDTH + row_B_id] = sum;
    }
  }


 return 0;
}

//////////////////////////// kernel function for v5 end /////////////////////////////

//////////////////////////// kernel function for v6 start/////////////////////////////

template <typename TA, typename TB>
int  __attribute__ ((noinline)) sparse_multiply_row_wise_v6(
      TA *A, TB *B, TB *Out, uint32_t A_HEIGHT, uint32_t A_WIDTH, uint32_t B_WIDTH)
{


  int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
  int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
  int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x;
  int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y;

  int num_threads = num_threads_x * num_threads_y;
  int tid = tid_y * num_threads_x + tid_x;
  
  int B_HEIGHT = A_WIDTH;
  TB sum;

  int mid_size = 0;

  int block_col_size = 250;

  float elem_A;

  // calculate the number of row and col blocks in matrix A and matrix B
  int B_num_blk_per_row = (B_WIDTH + block_col_size - 1) / block_col_size;

  // calculate the dimension of the last block size in each matrix
  int B_last_blk_col = (B_WIDTH % block_col_size == 0) ? block_col_size : (B_WIDTH % block_col_size);
  
  for (int inside_blk_id = 0; inside_blk_id < B_num_blk_per_row; inside_blk_id ++){
      mid_size = (inside_blk_id == B_num_blk_per_row - 1) ? B_last_blk_col : block_col_size;
      float data_B_row[mid_size];
      float tmp_sum[mid_size];
    for (int row_A_id = tid; row_A_id < A_HEIGHT; row_A_id += num_threads){
      memset(tmp_sum, 0, mid_size * sizeof(float));
      for (int sp_idx = sparse_mat_get_idx_ptr(A,row_A_id); sp_idx < sparse_mat_get_idx_ptr(A,row_A_id + 1); sp_idx++){
        int row_B_id = sparse_mat_get_col(A,sp_idx);
        memcpy(data_B_row, &B[row_B_id * B_WIDTH + inside_blk_id * block_col_size], sizeof(float)*mid_size);
        float elem_A = sparse_mat_get_val(A,sp_idx);
        for (int i = 0; i < mid_size; i++){
         tmp_sum[i] += elem_A * data_B_row[i];
        }
      }
      // write back to memory
      for (int i = 0; i < mid_size; i++){
        Out[row_A_id * B_WIDTH + inside_blk_id * block_col_size + i] = tmp_sum[i];
      }
    }
    
  }
  


 return 0;
}

//////////////////////////// kernel function for v6 end /////////////////////////////

//////////////////////////// kernel function for v7 start /////////////////////////////

template <typename TA, typename TB>
int  __attribute__ ((noinline)) sparse_multiply_row_wise_unrolling_v7(
      TA *A, TB *B, TB *Out, uint32_t A_HEIGHT, uint32_t A_WIDTH, uint32_t B_WIDTH)
{


  int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
  int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
  int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x;
  int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y;

  int num_threads = num_threads_x * num_threads_y;

  int tid = tid_y * num_threads_x + tid_x;
  
  int B_HEIGHT = A_WIDTH;
  TB sum;

  int mid_size = 0;

  int block_col_size = 16;

  float elem_A;
  int flag;
  int start_addr;

  // calculate the number of row and col blocks in matrix A and matrix B
  int B_num_blk_per_row = (B_WIDTH + block_col_size - 1) / block_col_size;

  // calculate the dimension of the last block size in each matrix
  int B_last_blk_col = (B_WIDTH % block_col_size == 0) ? block_col_size : (B_WIDTH % block_col_size);
  
  for (int inside_blk_id = 0; inside_blk_id < B_num_blk_per_row; inside_blk_id ++){
      mid_size = (inside_blk_id == B_num_blk_per_row - 1) ? B_last_blk_col : block_col_size;
      flag = (mid_size == block_col_size) ? 1 : 0;
      float data_B_row[mid_size];
      float tmp_sum[mid_size];
    for (int row_A_id = tid; row_A_id < A_HEIGHT; row_A_id += num_threads){
      memset(tmp_sum, 0, mid_size * sizeof(float));
      for (int sp_idx = sparse_mat_get_idx_ptr(A,row_A_id); sp_idx < sparse_mat_get_idx_ptr(A,row_A_id + 1); sp_idx++){
        int row_B_id = sparse_mat_get_col(A,sp_idx);
        start_addr = row_B_id * B_WIDTH + inside_blk_id * block_col_size;
        if(flag){
          unrolling_load(data_B_row, B, start_addr);
        }else{
          memcpy(data_B_row, &B[row_B_id * B_WIDTH + inside_blk_id * block_col_size], sizeof(float)*mid_size);
        }
        float elem_A = sparse_mat_get_val(A,sp_idx);
        if(flag){
          compute_unrolling(tmp_sum, elem_A, data_B_row);
        }else{
          for (int i = 0; i < mid_size; i++){
            tmp_sum[i] += elem_A * data_B_row[i];
          }
        }
      }
      // write back to memory
      int offset = row_A_id * B_WIDTH + inside_blk_id * block_col_size;
      for (int i = 0; i < mid_size; i++){
        Out[offset + i] = tmp_sum[i];
      }
      
    }
    
  }
  
 return 0;
}

//////////////////////////// kernel function for v7 end /////////////////////////////

//////////////////////////// kernel function for v8 start /////////////////////////////

template <typename TA, typename TB>
int  __attribute__ ((noinline)) sparse_multiply_row_wise_unrolling_v8(
      TA *A, TB *B, TB *Out, uint32_t A_HEIGHT, uint32_t A_WIDTH, uint32_t B_WIDTH)
{


  int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
  int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
  int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x;
  int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y;

  int num_threads = num_threads_x * num_threads_y;

  int tid = tid_y * num_threads_x + tid_x;
  
  int B_HEIGHT = A_WIDTH;
  TB sum;

  int mid_size = 0;

  int block_col_size = 2; // distribute the columns of dense matrix to tiles, block the rows of the dense matrix

  float elem_A;
  int start_addr;
  int col_inside_offset;
  int B_offset;
  int flag = 0;

  // calculate the number of row and col blocks in matrix A and matrix B
  int B_num_blk_per_row = (B_WIDTH + block_col_size - 1) / block_col_size;

  // calculate the dimension of the last block size in each matrix
  int B_last_blk_col = (B_WIDTH % block_col_size == 0) ? block_col_size : (B_WIDTH % block_col_size);

  for (int inside_blk_id = tid; inside_blk_id < B_num_blk_per_row; inside_blk_id += num_threads){
    mid_size = (inside_blk_id == B_num_blk_per_row - 1) ? B_last_blk_col : block_col_size;
    flag = (mid_size == block_col_size) ? 1 : 0;
    float tmp_sum[mid_size];
    float data_B_row[mid_size];
    col_inside_offset = inside_blk_id * block_col_size;
    for (int row_A_id = 0; row_A_id < A_HEIGHT; row_A_id ++){
      memset(tmp_sum, 0, mid_size * sizeof(float));
      for (int sp_idx = sparse_mat_get_idx_ptr(A,row_A_id); sp_idx < sparse_mat_get_idx_ptr(A,row_A_id + 1); sp_idx++){
        elem_A = sparse_mat_get_val(A, sp_idx);
        int row_B_id = sparse_mat_get_col(A,sp_idx);
        B_offset = row_B_id * B_WIDTH + col_inside_offset;
        memcpy(data_B_row, &B[B_offset], sizeof(float)*mid_size);
        if(flag){
          compute_unrolling_v8(tmp_sum, elem_A, data_B_row);
        }else{
          for(int blk_inside_idx = 0; blk_inside_idx < mid_size; blk_inside_idx ++){
            tmp_sum[blk_inside_idx] += elem_A * B[B_offset + blk_inside_idx];
          }
        } 
      }
      for (int i = 0; i < mid_size; i++){
        Out[row_A_id * B_WIDTH + col_inside_offset + i] = tmp_sum[i];
      }
    }
  }


 return 0;
}

//////////////////////////// kernel function for v8 end /////////////////////////////

//////////////////////////// kernel function for v9 start /////////////////////////////

template <typename TA, typename TB>
int  __attribute__ ((noinline)) sparse_multiply_row_wise_unrolling_v9(
      TA *A, TB *B, TB *Out, uint32_t A_HEIGHT, uint32_t A_WIDTH, uint32_t B_WIDTH)
{


  int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
  int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
  int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x; // 16
  int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y; // 8

  int num_threads = num_threads_x * num_threads_y;
 
  int tid = tid_y * num_threads_x + tid_x;
  
  int B_HEIGHT = A_WIDTH;
  TB sum;

  int mid_size = 0;

  int block_col_size = 16;

  float elem_A;
  int flag;
  int start_addr;

  // calculate the number of row and col blocks in matrix A and matrix B
  int B_num_blk_per_row = (B_WIDTH + block_col_size - 1) / block_col_size;

  // calculate the dimension of the last block size in each matrix
  int B_last_blk_col = (B_WIDTH % block_col_size == 0) ? block_col_size : (B_WIDTH % block_col_size);
  
  for (int inside_blk_id = tid_y; inside_blk_id < B_num_blk_per_row; inside_blk_id += num_threads_y){
      mid_size = (inside_blk_id == B_num_blk_per_row - 1) ? B_last_blk_col : block_col_size;
      flag = (mid_size == block_col_size) ? 1 : 0;
      float data_B_row[mid_size];
      float tmp_sum[mid_size];
    for (int row_A_id = tid_x; row_A_id < A_HEIGHT; row_A_id += num_threads_x){
      memset(tmp_sum, 0, mid_size * sizeof(float));
      for (int sp_idx = sparse_mat_get_idx_ptr(A,row_A_id); sp_idx < sparse_mat_get_idx_ptr(A,row_A_id + 1); sp_idx++){
        int row_B_id = sparse_mat_get_col(A,sp_idx);
        start_addr = row_B_id * B_WIDTH + inside_blk_id * block_col_size;
        if(flag){
          unrolling_load(data_B_row, B, start_addr);
        }else{
          memcpy(data_B_row, &B[row_B_id * B_WIDTH + inside_blk_id * block_col_size], sizeof(float)*mid_size);
        }
        float elem_A = sparse_mat_get_val(A,sp_idx);
        if(flag){
          compute_unrolling(tmp_sum, elem_A, data_B_row);
        }else{
          for (int i = 0; i < mid_size; i++){
            tmp_sum[i] += elem_A * data_B_row[i];
          }
        }
      }
      // write back to memory
      int offset = row_A_id * B_WIDTH + inside_blk_id * block_col_size;
      for (int i = 0; i < mid_size; i++){
        Out[offset + i] = tmp_sum[i];
      }
      
    }
    
  }
  


 return 0;
}

//////////////////////////// kernel function for v9 end /////////////////////////////

//////////////////////////// kernel function for v10 start /////////////////////////////

// Included in the kernel code

//////////////////////////// kernel function for v10 end /////////////////////////////

#endif //__SPARSE_MULTIPLY_HPP_ALL_FUNCTION
