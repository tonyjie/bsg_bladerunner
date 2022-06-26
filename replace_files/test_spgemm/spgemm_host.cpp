#include "spgemm_host.hpp"
#include <string>
#include <unordered_set>
#include <bsg_manycore_regression.h>
#define ALLOC_NAME "default_allocator"
// All matrices are square
#define A_HEIGHT    30
#define A_WIDTH     30
#define B_WIDTH     30
#define OUT_WIDTH   B_WIDTH
#define NNZ_PER_ROW 8
#define tiles_num   128

// Print matrix A (M x N).10
template <typename T>
void matrix_print(T *A, uint64_t M, uint64_t N) {
    for (uint64_t y = 0; y < M; y ++) {
        for (uint64_t x = 0; x < N; x ++) {
            printf("%f ", A[y * N + x]);
        }
        printf("\n");
    }
}

/////////////////////////////////////////////////
// CSR sparse matrix data structure and utilities
/////////////////////////////////////////////////
typedef float data_t;

using namespace std;

typedef struct sparse_mat_t {
  // number of non-zero
  int nnz;

  // the dimensions of the matrix
  // m is y, n is x
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


/////////////////////////////////////////////////
// sparse matrix utilities on host
/////////////////////////////////////////////////
sparse_mat_t *create_from_sparse(data_t *nz_data, int *idx_ptrs,
                                 int *nz_cols, int m, int n, int nnz) {
  sparse_mat_t *mat = (sparse_mat_t*)malloc(sizeof(sparse_mat_t));
  mat->m = m;
  mat->n = n;
  mat->nnz = nnz;
  // hardcopy all of the arrays
  mat->idx_ptrs = (int*)malloc(sizeof(int) * m + 1);
  for (int i = 0; i < m + 1; i++) {
    mat->idx_ptrs[i] = idx_ptrs[i];
  }
  mat->nz_cols = (int*)malloc(sizeof(int) * nnz);
  mat->nz_data = (data_t*)malloc(sizeof(data_t) * nnz);
  for (int i = 0; i < nnz; i++) {
    mat->nz_cols[i] = nz_cols[i];
    mat->nz_data[i] = (data_t)nz_data[i];
  }
  return mat;
}

void destroy_sparse_mat(sparse_mat_t *mat) {
  free(mat->idx_ptrs);
  free(mat->nz_cols);
  free(mat->nz_data);
  free(mat);
}

int size_of_device_fields() {
  return sizeof(uint32_t) * 6;
}

// ret a pointer to the struct
eva_t sparse_mat_memcpy_to(hb_mc_device_t *device, sparse_mat_t *mat) {
  int err;
  // malloc the backing arrays
  eva_t nz_data_dev, nz_cols_dev, idx_ptrs_dev;
  const int num_bytes_data = sizeof(uint32_t) * mat->nnz; 
  const int num_bytes_cols = num_bytes_data;
  const int num_bytes_ptrs = sizeof(uint32_t) * (mat->m + 1);
  err  = hb_mc_device_malloc(device, num_bytes_data, &nz_data_dev);
  err |= hb_mc_device_malloc(device, num_bytes_cols, &nz_cols_dev);
  err |= hb_mc_device_malloc(device, num_bytes_ptrs, &idx_ptrs_dev);
  // copy the backing arrays to the device
  /*
  err |= hb_mc_device_memcpy (device, (void*)((intptr_t)nz_data_dev), (void*)(mat->nz_data),
                        num_bytes_data, HB_MC_MEMCPY_TO_DEVICE);
  err |= hb_mc_device_memcpy (device, (void*)((intptr_t)nz_cols_dev), (void*)(mat->nz_cols),
                        num_bytes_cols, HB_MC_MEMCPY_TO_DEVICE);
  err |= hb_mc_device_memcpy (device, (void*)((intptr_t)idx_ptrs_dev), (void*)(mat->idx_ptrs),
                        num_bytes_ptrs, HB_MC_MEMCPY_TO_DEVICE);
  */
  printf("start copying sparse matrix to device\n");
  hb_mc_dma_htod_t nz_data_dma = {nz_data_dev, (void*)(mat->nz_data), num_bytes_data};
  hb_mc_dma_htod_t nz_cols_dma = {nz_cols_dev, (void*)(mat->nz_cols), num_bytes_cols};
  hb_mc_dma_htod_t idx_ptrs_dma = {idx_ptrs_dev, (void*)(mat->idx_ptrs), num_bytes_ptrs};
  err |= hb_mc_device_dma_to_device(device, &nz_data_dma, 1);
  err |= hb_mc_device_dma_to_device(device, &nz_cols_dma, 1);
  err |= hb_mc_device_dma_to_device(device, &idx_ptrs_dma, 1);
  // set the pointers in the struct
  // explicitly assign pointer values to the struct
  int mat_fields[6];
  mat_fields[0] = mat->nnz;
  mat_fields[1] = mat->m; //row
  mat_fields[2] = mat->n; //column
  mat_fields[3] = nz_data_dev;
  mat_fields[4] = idx_ptrs_dev;
  mat_fields[5] = nz_cols_dev;
  // allocate and copy the struct to the device
  eva_t struct_dev;
  const int num_bytes_struct = size_of_device_fields();
  err |= hb_mc_device_malloc(device, num_bytes_struct, &struct_dev);
  /*
  err |= hb_mc_device_memcpy (device, (void*)((intptr_t)struct_dev), (void*)(mat_fields),
                        num_bytes_struct, HB_MC_MEMCPY_TO_DEVICE);
  if (err != HB_MC_SUCCESS) {
    printf("ERROR: failed to copy sparse mat to device\n");
  }
  */
  hb_mc_dma_htod_t struct_dev_dma = {struct_dev, (void*)(mat_fields), num_bytes_struct};
  err |= hb_mc_device_dma_to_device(device, &struct_dev_dma, 1);

  if (err != HB_MC_SUCCESS) {
    printf("ERROR: failed to copy sparse mat to device\n");
  }

  return struct_dev;
}

void sparse_mat_memcpy_from(hb_mc_device_t *device, eva_t struct_dev, sparse_mat_t *mat) {
  int err;
  const int num_bytes_struct = size_of_device_fields(); 
  int mat_fields[6];
  err = hb_mc_device_memcpy(device, (void*)(mat_fields), (void*)((intptr_t)struct_dev),
                            num_bytes_struct, HB_MC_MEMCPY_TO_HOST);
  mat->nnz = mat_fields[0];
  mat->m   = mat_fields[1];
  mat->n   = mat_fields[2];
  eva_t nz_data_dev = mat_fields[3];
  eva_t idx_ptrs_dev = mat_fields[4];
  eva_t nz_cols_dev = mat_fields[5];
  // track down the data arrays
  const int num_bytes_data = sizeof(uint32_t) * mat->nnz;
  const int num_bytes_cols = num_bytes_data;
  const int num_bytes_ptrs = sizeof(uint32_t) * (mat->m + 1);
  mat->nz_data = (data_t*)malloc(num_bytes_data);
  mat->nz_cols = (int*)malloc(num_bytes_cols);
  mat->idx_ptrs = (int*)malloc(num_bytes_ptrs);
  // copy the backing arrays to the host
  err |= hb_mc_device_memcpy (device, (void*)(mat->nz_data), (void*)((intptr_t)nz_data_dev),
                         num_bytes_data, HB_MC_MEMCPY_TO_HOST);
  err |= hb_mc_device_memcpy (device, (void*)(mat->nz_cols), (void*)((intptr_t)nz_cols_dev),
                        num_bytes_cols, HB_MC_MEMCPY_TO_HOST);
  err |= hb_mc_device_memcpy (device, (void*)(mat->idx_ptrs), (void*)((intptr_t)idx_ptrs_dev),
                        num_bytes_ptrs, HB_MC_MEMCPY_TO_HOST);
  if (err != HB_MC_SUCCESS) {
    printf("ERROR: failed to copy sparse mat to host\n");
  }
  
}

// Host spgemm (to compare results)
// void spgemm_reference(sparse_mat_t *A, sparse_mat_t *B, data_t *Out, int M, int N, int K) { // M is y of A, N is x of A, K is x f B
//   for (int row_A_id = 0; row_A_id < M; row_A_id++) {
//     for (int j = 0; j < K; j++) {
//       Out[row_A_id * K + j] = 0;
//     }
//   }
//   for (int row_A_id = 0; row_A_id < M; row_A_id++) {
//     for (int col_B_id = 0; col_B_id < K; col_B_id ++){
//       for (int i = sparse_mat_get_idx_ptr(A, row_A_id); i < sparse_mat_get_idx_ptr(A, row_A_id + 1); i++){
//         data_t elem_A = sparse_mat_get_val(A, i);
//         int row_B_id = sparse_mat_get_col(A, i);
//         int B_id = sparse_mat_get_idx_ptr(B, row_B_id);
//         data_t elem_B = sparse_mat_get_val(B, B_id);
//         if (elem_B != 0){
//           Out[row_A_id * K + col_B_id] += elem_A * elem_B;
//         }
//       }
//     }
//   }
// }
void spgemm_reference(sparse_mat_t *A, sparse_mat_t *B, data_t *Out, int M, int N, int K) { // M is y of A, N is x of A, K is x f B
  for (int row_A_id = 0; row_A_id < M; row_A_id++) {
    for (int j = 0; j < K; j++) {
      Out[row_A_id * K + j] = 0;
    }
  }
  data_t sum = 0;
  for (int row_A_id = 0; row_A_id < M; row_A_id++) {
    for (int col_B_id = 0; col_B_id < K; col_B_id ++){
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
      Out[row_A_id * K + col_B_id] = sum;
    }
  }
}

// Compute the sum of squared error between matrix A and B (M x N)
double check_dense(const data_t *A, const data_t *B, int M, int N) {
  double sum = 0;
  for (int x = 0; x < M * N; x++) {
    
    printf("index = %d; reference = %2.6f , %2.6f \n", x, A[x], B[x]);
    data_t diff = A[x] - B[x];
    if(std::isnan(diff)){
        return diff;
    }
    sum += diff * diff;
  }
  return sum;
}

// takes csr matrix as input and returns a dense matrix
void csr_to_dense(sparse_mat_t* A, data_t* out) {
  printf("m = %d\n",A->m); 
  int m = A->m;
  printf("n = %d\n",A->n); 
  int n = A->n;

  // initialize all values to 0
  for (int i = 0; i < A->m * A->n; i++) {
    out[i] = 0;
  }

  // data_t out[m * n];
  for (int i = 0; i < A->m; i++) {
    int row =i;
      
    for (int j=A->idx_ptrs[i]; j<A->idx_ptrs[i+1];j++){
      int col = A->nz_cols[j];
      printf("row,col = %d,%d, index = %d; value is %2.6f\n",row,col, n*row + col, A->nz_data[j]); 
      out[n*row + col] = A->nz_data[j];
    } 
  }
}

sparse_mat_t* dense_to_csr(const data_t(*A)[MAT_SIZE], int m, int n) {
  int nnz = 0;
  int row_ptr = 0;
  int idx_ptrs[m+1];
  idx_ptrs[0] = 0;
  for (int i = 0; i < m; i++) { 
    for (int j = 0; j < n; j++) {
      if(A[i][j] != 0) {
        nnz++;
      }
    }
  }
  int nz_cols[nnz];
  int nz_data[nnz];
  int nz = 0;
  for(int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if(A[i][j] != 0) {
        nz_cols[nz] = j;
        nz_data[nz] = A[i][j];
        nz = nz + 1;
      }
    } 
    idx_ptrs[i+1] = nz;
  }

  sparse_mat_t *mat = (sparse_mat_t*)malloc(sizeof(sparse_mat_t));
  mat->m = m;
  mat->n = n;
  mat->nnz = nnz;
  // hardcopy all of the arrays
  mat->idx_ptrs = (int*)malloc(sizeof(int) * m + 1);
  for (int i = 0; i < n + 1; i++) {
           mat->idx_ptrs[i] = idx_ptrs[i];
       }
  mat->nz_cols = (int*)malloc(sizeof(int) * nnz);
  mat->nz_data = (data_t*)malloc(sizeof(data_t) * nnz);
  for (int i = 0; i < nnz; i++) {
    mat->nz_cols[i] = nz_cols[i];
    mat->nz_data[i] = (data_t)nz_data[i];
  }
  printf("nnz is: %d\n", nnz);
  for(int i = 0; i < m+1; i++) {
    printf("idx_ptrs[%d] is %d\n", i, idx_ptrs[i]);
  } 
  return mat;
}

sparse_mat_t *read_sparse_matrix(string filename) {
  printf("start reading the sparse matrix\n");
  int m = 0, k = 0;
  int nnz = 0;

  int section = 0;
  string line;
  ifstream rf(filename);
  if (rf.is_open()){
    while (getline (rf, line)) {
      if(line.compare("shape:") == 0){
	      getline (rf, line);
	      string token = line.substr(1, line.length() - 2);
	      m = stoi(token.substr(0, token.find(", ")));
        printf("the value of m = %d\n", m);
	      k = stoi(token.substr(token.find(", ") + 2, token.length() - (token.find(", ") + 2)));
        printf("the value of k = %d\n", k);
      }
      if(line.compare("nnz:") == 0){
	      getline (rf, line);
	      nnz = stoi(line);
        printf("the value of nnz = %d\n", nnz);
	      break;
      }
    }
  }
  else{
    printf("cannot open file\n");
  }
  
  int idx_ptrs[m + 1];
  data_t nz_data[nnz];
  int nz_cols[nnz];
  
  while (getline (rf, line)) {
    if(line.compare("indptr:") == 0){
      int idx = 0;
      while(idx < m + 1){
	      getline (rf, line);
	      idx_ptrs[idx] = stoi(line);
	      idx++;
      }
    }
    if(line.compare("indices:") == 0){
      int idx = 0;
      while(idx < nnz){
	      getline (rf, line);
	      nz_cols[idx] = stoi(line);
	      idx++;
      }
    }
    if(line.compare("values:") == 0){
      int idx = 0;
      while(idx < nnz){
	      getline (rf, line);
	      nz_data[idx] = stof(line);
	      idx++;
      }
    }
  }

  // Close the file
  rf.close();
  return create_from_sparse(nz_data, idx_ptrs, nz_cols, m, k, nnz);
}


void matrix_transpose (data_t *A, data_t *B, int M, int N) {
  for (int y = 0; y < M; y ++) {
    for (int x = 0; x < N; x ++) {
      B[x * M + y] = A[y * N + x];
    }
  }
}


sparse_mat_t *CSR_transpose(sparse_mat_t *A){
  data_t * nz_data = (data_t*)malloc(sizeof(data_t) * A->nnz);
  int * idx_ptrs = (int*)malloc(sizeof(int) * (A->n + 1));
  int * nz_cols = (int*)malloc(sizeof(int) * A->nnz);
  int * curr = (int*)malloc(sizeof(int) * A->n);
	
  sparse_mat_t *result = create_from_sparse(nz_data, idx_ptrs, nz_cols, A->n, A->m, A->nnz);
	
  for (int i = 0; i < A->nnz; i++) {
    result->idx_ptrs[A->nz_cols[i] + 1] = result->idx_ptrs[A->nz_cols[i] + 1] + 1 ; 
  }

  for (int i = 1; i < sizeof(result->idx_ptrs); i++) { 
    result->idx_ptrs[i] = result->idx_ptrs[i] + result->idx_ptrs[i - 1];
  }
	
  for (int i = 0; i < A->m; i++) {
    for (int j = A->idx_ptrs[i]; j < A->idx_ptrs[i + 1]; j++) {
      int new_index = result->idx_ptrs[A->nz_cols[j]]+curr[A->nz_cols[j]];
      curr[A->nz_cols[j]] = curr[A->nz_cols[j]]+1; 
      result->nz_data[new_index] = A->nz_data[j];
      result->nz_cols[new_index] = i;
    }
  }
  free(curr); 
  destroy_sparse_mat(A); 
  return result; 
}

int test_spgemm(int argc, char *argv[]) {
  char *bin_path, *test_name;
  struct arguments_path args = {NULL, NULL};
  argp_parse(&argp_path, argc, argv, 0, 0, &args);
  bin_path = args.path;
  test_name = args.name;

  printf("Running sparse matrix sparse matrix.\n\n");
  int err;
  
  ///////////////////////////////////// sparse matrix data source //////////////////////////////////////

  // (1) generate the custom sparse matrix using the generator (begin here)
  
  int m = A_HEIGHT, n = A_WIDTH, k = B_WIDTH;
  int nnz_per_row = NNZ_PER_ROW;
  int nnz = m * nnz_per_row;
  int idx_ptrs[m + 1];
  data_t nz_data[nnz];
  int nz_cols[nnz];
  for (int i = 0; i < m + 1; i++) {
    idx_ptrs[i] = i * nnz_per_row;
  }
  for (int i = 0; i < nnz; i++) {
    nz_data[i] = (float)(rand()%100 + 200) / 100.0f;
    nz_cols[i] = i % nnz_per_row;
  }
  
  sparse_mat_t *sparse_matrix1 = create_from_sparse(nz_data, idx_ptrs, nz_cols, m, n, nnz);
  
  // (1) generate the custom sparse matrix using the generator (end here)

  // (2) read in the 10x500 sparse matrix txt file (begin here)
  
  // int m = 0, n = 0, k = 0;
  // sparse_mat_t *sparse_matrix1 = read_sparse_matrix("testcsr.txt");

  // m = sparse_matrix1->m;
  // n = sparse_matrix1->n;
  // k = B_WIDTH;
  
  // (2) read in the 10x500 sparse matrix txt file (end here)
  
  // (3) read in the 1000x1000 dense matrix dat file, transform it to scr format as sparse matrix (begin here)
  
  // sparse_mat_t *sparse_matrix1 = dense_to_csr(input_A, MAT_SIZE, MAT_SIZE); 
  // int m = sparse_matrix1->m;
  // int n = sparse_matrix1->n;
  // int k = B_WIDTH;

  // (3) read in the 1000x1000 dense matrix dat file, transform it to scr format as sparse matrix (end here)
  ///////////////////////////////////// sparse matrix data source end //////////////////////////////////////

  
  //There should be other ways to generate the second Sparse Matix. 
  nnz_per_row = NNZ_PER_ROW;
  nnz = n * nnz_per_row;
  int idx_ptrs2[n + 1];
  data_t nz_data2[nnz];
  int nz_cols2[nnz];
  for (int i = 0; i < n + 1; i++) {
    idx_ptrs2[i] = i * nnz_per_row;
  }
  for (int i = 0; i < nnz; i++) {
    nz_data2[i] = (float)(rand()%100 + 200) / 100.0f;
    nz_cols2[i] = i % nnz_per_row;
  }
  
  sparse_mat_t *sparse_matrix2 = create_from_sparse(nz_data2, idx_ptrs2, nz_cols2, n, k, nnz);

  // Number of tiles in y direction is set as 8 below for v4 kernel

  // Calculate number of rows each tile work on (rows_per_tile = number of rows / num tiles)
  int rows_per_tile = ceil((float) sparse_matrix1->m/(float) tiles_num);

  // Create and Store num nonzero rows in array (tile_boundary)
  int tile_boundary[tiles_num+1];
  for (int i=0; i<tiles_num+1; i++){
    tile_boundary[i]=0;
  }

  // counters to track when we should go to next tile
  int curr_tile = 1;
  int curr_row = 0;
  int A_nnz_rows = 0;
  
  for (int i = 1; i < sparse_matrix1->m + 1; i++) {
    if (sparse_matrix1->idx_ptrs[i-1] != sparse_matrix1->idx_ptrs[i]) {
      // total number of nonzero rows up to and including the current tile
      tile_boundary[curr_tile] = ++A_nnz_rows;
    }

    curr_row++;
    // go to next tile based on the pre-calculated number of rows in each tile
    if (curr_row >= rows_per_tile) {
      curr_tile++;
      curr_row = 0;
    }
  }
  
  for (int i = curr_tile; i <tiles_num+1; i++ ){
    tile_boundary[i]= tile_boundary[i-1]; 
  }

  for (int i = 0; i <tiles_num+1; i++ ){
    printf("tile_boundary=%d",tile_boundary[i] ); 
  }

  // determining how many nonzero columns are in B
  unordered_set<int> B_nnz_cols;
  for (int i = 0; i < nnz; i++)
    B_nnz_cols.insert(sparse_matrix2->nz_cols[i]);

  // in the worst case, nnz(Out) = A_nnz_rows * B_nnz_cols
  int nnz_max = A_nnz_rows * B_nnz_cols.size();

  for (int i=0; i < tiles_num+1; i++){
    tile_boundary[i] = tile_boundary[i] * B_nnz_cols.size();
  }

  // set up the sparse output assuming this nnz
  int idx_ptrs_out[m + 1];
  for (int i=0; i<m+1; i++){
    idx_ptrs_out[i]=0; 
  }
  
  data_t nz_data_out[nnz_max];
   for (int i=0; i<nnz_max; i++){
    nz_data_out[i]=0; 
  }
  int nz_cols_out[nnz_max]; 
   for (int i=0; i<nnz_max; i++){
    nz_cols_out[i]=0; 
  }
  sparse_mat_t *sparse_matrix_out = create_from_sparse(nz_data_out, idx_ptrs_out, nz_cols_out, m, k, nnz_max);

  data_t *dense_matrix;
  data_t out_matrix_input[m * k];

  data_t Out_reference[m * k];

  for (int i = 0; i < m * k; i++){
    out_matrix_input[i] = 0;
  }
	
  // Generate the known-correct results on the host
//  if(!strcmp("v1", test_name)){
//    sparse_matrix2 = CSR_transpose(sparse_matrix2);
//  }

  // have to update for when output is sparse matrix
  spgemm_reference(sparse_matrix1, sparse_matrix2, Out_reference, m, n, k);
	
  printf("start creating device\n");
  hb_mc_device_t device;
  err = hb_mc_device_init(&device, test_name, 0);
  if (err != HB_MC_SUCCESS) {
    printf("failed to initialize device.\n");
    return err;
  }

  printf("start doing hb_mc_device_program_init\n");
  err = hb_mc_device_program_init(&device, bin_path, "default_allocator", 0);
  if (err != HB_MC_SUCCESS) {
    printf("failed to initialize program.\n");
    return err;
  }

  /*---------------------------------------------------------------------
  * Copy the the matrices to the hammerblade
  *-------------------------------------------------------------------*/
  printf("start sparse_mat_memcpy_to\n");
  eva_t A_matrix_dev;
  A_matrix_dev = sparse_mat_memcpy_to(&device, sparse_matrix1);

  printf("start sparse_mat_memcpy_to\n");
  eva_t B_matrix_dev;
  B_matrix_dev = sparse_mat_memcpy_to(&device, sparse_matrix2);
  /*
  err = hb_mc_device_memcpy(&device, (void*)((intptr_t)B_matrix_dev), (void*)dense_matrix,
                            k * n * sizeof(data_t), HB_MC_MEMCPY_TO_DEVICE);
  if (err != HB_MC_SUCCESS) {
    printf("failed to copy memory to device.\n");
    return err;
  }
  */

  eva_t Out_matrix_dev;
  eva_t tile_boundary_dev;
  eva_t tile_nnz_dev;
  eva_t col_temp_dev;
  eva_t val_temp_dev;

  // v0-v3, output is dense matrix
//  if (strcmp("v4", test_name)) {
//    err = hb_mc_device_malloc(&device, m * k * sizeof(data_t), &Out_matrix_dev);
//    if (err != HB_MC_SUCCESS) {
//      printf("failed to allocate memory on device.\n");
//      return err;
//    }
//    hb_mc_dma_htod_t out_matrix_dma = {Out_matrix_dev, (void*)out_matrix_input, m * k * sizeof(data_t)};
//    err = hb_mc_device_dma_to_device(&device, &out_matrix_dma, 1);
//    if (err != HB_MC_SUCCESS) {
//      printf("failed to copy memory to device.\n");
//      return err;
//    }
//  }

  // for v4, output should be sparse matrix
//  else {
    Out_matrix_dev = sparse_mat_memcpy_to(&device, sparse_matrix_out);

    // copying tile_boundary array to device (tracks nnz upper bound for each tile)
//    int err;
    int rc;
    const int num_bytes_tile_boundary = sizeof(int) * (tiles_num + 1);
    err  = hb_mc_device_malloc(&device, num_bytes_tile_boundary, &tile_boundary_dev);
    if (err != HB_MC_SUCCESS) {
      printf("failed to allocate memory on device for tile_boundary.\n");
    }
    printf("start copying tile boundary array to device\n");
    hb_mc_dma_htod_t tile_boundary_dma = {tile_boundary_dev, (void*)(tile_boundary), num_bytes_tile_boundary};
    err |= hb_mc_device_dma_to_device(&device, &tile_boundary_dma, 1);
    // void *dst = (void *) ((intptr_t)tile_boundary_dev);
    // void *src = (void *) &tile_boundary[0];
    // rc = hb_mc_device_memcpy(&device, dst, src, num_bytes_tile_boundary, HB_MC_MEMCPY_TO_DEVICE);

    // copying tile_nnz array to device (used for each tile to record its actual nnz in kernel code)
    int tile_nnz[tiles_num+1];
    for (int i=0; i<tiles_num+1; i++){
      tile_nnz[i]=0; 
    }
    const int num_bytes_tile_nnz = sizeof(int) * (tiles_num + 1);
    err  = hb_mc_device_malloc(&device, num_bytes_tile_nnz, &tile_nnz_dev);
    printf("start copying tile_nnz array to device\n");
    hb_mc_dma_htod_t tile_nnz_dma = {tile_nnz_dev, (void*)(tile_nnz), num_bytes_tile_nnz};
    err |= hb_mc_device_dma_to_device(&device, &tile_nnz_dma, 1);

    // Allocate temporary arrays to store nz values and column indexes 
    int col_idx_temp[nnz_max]; 
    for (int i=0; i<nnz_max; i++){
      col_idx_temp[i]=0; 
    }
    const int num_bytes_col_temp = sizeof(int) * nnz_max; 
    err  = hb_mc_device_malloc(&device, num_bytes_col_temp, &col_temp_dev);
    printf("start copying temporary column index array to device\n");
    hb_mc_dma_htod_t col_temp_dma = {col_temp_dev, (void*)(col_idx_temp), num_bytes_col_temp};
    err |= hb_mc_device_dma_to_device(&device, &col_temp_dma, 1);

    data_t val_temp[nnz_max]; 
    const int num_bytes_val_temp = sizeof(data_t) * nnz_max; 
    err  = hb_mc_device_malloc(&device, num_bytes_val_temp, &val_temp_dev);
    printf("start copying temporary nnz value array to device\n");
    hb_mc_dma_htod_t val_temp_dma = {val_temp_dev, (void*)(val_temp), num_bytes_val_temp};
    err |= hb_mc_device_dma_to_device(&device, &val_temp_dma, 1);
//  }

  hb_mc_dimension_t grid_dim = { .x = 0, .y = 0};
  hb_mc_dimension_t tg_dim = { .x = 0, .y = 0 };
  hb_mc_dimension_t block_size = { .x = 0, .y = 0 };
  
//  if(!strcmp("v0", test_name)){
//    grid_dim = { .x = 1, .y = 1}; // v0 is inner product, serves as baseline
//    tg_dim = { .x = 4, .y = 4 };  
//  }
  // v1 will be same as v0, but transpose the second sparse matrix
//  else if(!strcmp("v2", test_name)){
//    grid_dim = { .x = 1, .y = 1}; // v2 is outer product
//    tg_dim = { .x = 4, .y = 4 };  
//  }
  // v4
//  else if(!strcmp("v4", test_name)){
    grid_dim = { .x = 1, .y = 1};
    tg_dim = { .x = 16, .y = 8 }; 
//  }
//  else {
//    bsg_pr_test_err("Invalid version provided!.\n");
//    return HB_MC_INVALID;
//  }
  /*---------------------------------------------------------------------
  * Prepare list of input arguments for kernel
  *---------------------------------------------------------------------*/
  // For v4, there are 10 arguments input

  uint32_t cuda_argv_dense[6] = {A_matrix_dev, B_matrix_dev, Out_matrix_dev, m, n, k};
  uint32_t cuda_argv_sparse[10] = {A_matrix_dev, B_matrix_dev, Out_matrix_dev,m, n, k, tile_boundary_dev, col_temp_dev, val_temp_dev, tile_nnz_dev};

  int cuda_argc; 
//  if(strcmp("v4", test_name)){
//    cuda_argc = 6;
//  }

//  else {
    cuda_argc = 10;
//  }

  printf("hb_mc_kernel_enqueue\n");
//  if(strcmp("v4", test_name)){
//    err = hb_mc_kernel_enqueue(&device, grid_dim, tg_dim, "kernel_spgemm", cuda_argc, cuda_argv_dense);
//  }

//  else {
    err = hb_mc_kernel_enqueue(&device, grid_dim, tg_dim, "kernel_spgemm", cuda_argc, cuda_argv_sparse);
//  }

  if (err != HB_MC_SUCCESS) {
    printf("failed to hb_mc_kernel_enqueue.\n");
    return err;
  }

  printf("hb_mc_device_tile_groups_execute\n");
  err = hb_mc_device_tile_groups_execute(&device);
  if (err != HB_MC_SUCCESS) {
    printf("failed to execute tile groups.\n");
    return err;
  }

  /*--------------------------------------------------------------------
  * Copy result matrix back from device DRAM into host memory.
  *-------------------------------------------------------------------*/
  
  data_t Out_host[m * k];
  printf("Create Out_host_CSR.\n");

  int idx_ptrs_out_host[m + 1];
  for (int i=0; i<m+1; i++){
    idx_ptrs_out_host[i]=0; 
  }
  data_t nz_data_out_host[nnz_max];
  for (int i=0; i<nnz_max; i++){
    nz_data_out_host[i]=0; 
  }
  int nz_cols_out_host[nnz_max];  
  for (int i=0; i<nnz_max; i++){
    nz_cols_out_host[i]=0; 
  }
  sparse_mat_t *Out_host_CSR = create_from_sparse(nz_data_out_host, idx_ptrs_out_host, nz_cols_out_host, m, k, nnz_max);

  //V0-3
//  if(strcmp("v4", test_name)){
//    err = hb_mc_device_memcpy(&device, (void *)Out_host, (void *)((intptr_t)Out_matrix_dev),
//                            m * k * sizeof(data_t),
//                            HB_MC_MEMCPY_TO_HOST);
//    if (err != HB_MC_SUCCESS) {
//      bsg_pr_test_err("failed to copy memory from device.\n");
//      return err;
//    }

//    printf("\n");
//    printf("---------------------------------Matrix B--------------------------------------\n"); // for non-transposeed B
//    matrix_print(dense_matrix, n, k);
//    printf("---------------------------------Matrix C--------------------------------------\n");
//    matrix_print(Out_host, m, k);
//    printf("-----------------------------Reference result--------------------------------------\n");
//    matrix_print(Out_reference, m, k);
//  }

  //v4 
//  else {
    printf("Copy device to Host.\n");
    sparse_mat_memcpy_from(&device, Out_matrix_dev, Out_host_CSR);
    
    for (int i= 0; i< Out_host_CSR->idx_ptrs[11]; i++){
      printf("col_index = %d, data = %d, # = %d\n", Out_host_CSR->nz_cols[i],Out_host_CSR->nz_data[i], i); 
    }
   
    printf("csr to dense.\n");
    csr_to_dense(Out_host_CSR, Out_host); 
//  }

  
  double tolerance = 0.1;
  double error; 

  printf("check_dense.\n");
  error = check_dense(Out_reference, Out_host, m, k);

  //cout << error << endl;
  if (error > tolerance) {
    bsg_pr_test_err(BSG_RED("Mismatch. Error: %f\n"), error);
    return HB_MC_FAIL;
  }
  bsg_pr_test_info(BSG_GREEN("Match.\n"));

  /*--------------------------------------------------------------------
  * Freeze the tiles and memory manager cleanup.
  *-------------------------------------------------------------------*/
  err = hb_mc_device_finish(&device);
  if (err != HB_MC_SUCCESS) {
    printf("failed to de-initialize device.\n");
    return err;
  }

  destroy_sparse_mat(sparse_matrix1);
  destroy_sparse_mat(sparse_matrix2);
  destroy_sparse_mat(sparse_matrix_out);

  return HB_MC_SUCCESS;
}

declare_program_main("test_spgemm", test_spgemm);
