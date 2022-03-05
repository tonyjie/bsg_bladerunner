#include "smith_waterman_host.hpp"
#include <string>
#include <unordered_set>
#include <bsg_manycore_regression.h>
#define ALLOC_NAME "default_allocator"

typedef float data_t;

using namespace std;


void reference_smith_waterman(std::string &seqa, std::string &seqb, int* matrix, int m, int n, const int match_score, const int mismatch_score, const int gap) {
  const auto mm = [&](const char a, const char b){ return (a==b)?match_score:mismatch_score; };
  for(int i = 1; i < m+1; i++) {
    for(int j = 1; j < n+1; j++) {
      int up = matrix[(i-1)*(n+1)+j] + gap;
      int left = matrix[i*(n+1)+j-1] +gap;
      int diag = matrix[(i-1)*(n+1)+j-1] + mm(seqb[i-1], seqa[j-1]);
      int temp = std::max(up, left);
      temp = std::max(temp, 0);
      matrix[i*(n+1)+j] = std::max(temp, diag);
    }
  } 
}

int check_result_correctness(const int *A, const int *B, int m, int n) {
  int sum = 0;
  for (int x = 0; x < (m+1) * (n+1); x++) {
    int diff = A[x] - B[x];
    sum += diff * diff;
  }
  return sum;
}

int test_smith_waterman (int argc, char *argv[]) {

  char *bin_path, *test_name;
  struct arguments_path args = {NULL, NULL};
  argp_parse(&argp_path, argc, argv, 0, 0, &args);
  bin_path = args.path;
  test_name = args.name;

  std::string seqa = "GCATCGACAAGCGTTTCCATCCGGTTTCCGGTATAGAGATGCAGGGTCATGTACGCCGGAGGGATGATTCGGGAGTTTCCTGGAAGGCGTCTATAGCTTCCATGGCCTTGCGGCGGAATCCGAGCGGATCGCCGATACCCGGTATGGGGGTTATCCCTCCTCCGGTGCCGTTGATGCTGAGCGTTCCGAAACCAAGAATTCGTCCAAGCACCCCCTGTTCCACATGGAAGCTCTCGACCTTGCTGTGGTTCAGTTCGATGGTGCTGCGACGGATGAAGCCGAATTTCGCTATGATCCGCTTCGACGTCACGGCAAGCTCGGTGGACTGGCGCTTGACGAACGCCTCACCCGTAAACCATAATCCGGCCACCAG";
  std::string seqb = "CGAATTTCGCTATGATCCGCTTCGACGTCACGGCAAGCTCGGTGGACTGGCGCTTGACGAACGCCTCACCCGTAAACCATAATTCGGCCACCAGAATGACGACTCCGGTATACCAGGCAGCCGAAGAAAGTTCCGGATTGCCATCGGGCG";
  std::string seqc = "ABCDE";
  int match = 1;
  int mismatch = -1;
  int gap = -2;
  printf("The size of the tested string is: %d\n", seqc.size());	

  printf("start creating device\n");
  hb_mc_device_t device;
  int err;
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

  int m = (int)seqb.size();
  int n = (int)seqa.size();
  int matrix_size = (m+1) * (n+1);

  int* cpu_matrix = (int*)malloc(matrix_size * sizeof(int));
  int* hb_matrix = (int*)malloc(matrix_size * sizeof(int));

  for(int i = 0; i < m+1; i++) {
    cpu_matrix[i*(n+1)] = 0;
    hb_matrix[i*(n+1)] = 0;
  }

  for(int j = 0; j < n+1; j++) {
    cpu_matrix[j] = 0;
    hb_matrix[j] = 0;
  }

  reference_smith_waterman(seqa, seqb, cpu_matrix, m, n, match, mismatch, gap);
  /*---------------------------------------------------------------------
  * Copy the the matrices to the hammerblade
  *-------------------------------------------------------------------*/

  eva_t seqa_dev;
  eva_t seqb_dev;
  eva_t matrix_dev;

  const int seqa_length = sizeof(char) * seqa.size();
  err  = hb_mc_device_malloc(&device, seqa_length, &seqa_dev);
  if (err != HB_MC_SUCCESS) {
    printf("failed to allocate memory on device for seqa.\n");
  }
  printf("start copying seqa to device\n");
  hb_mc_dma_htod_t seqa_dma = {seqa_dev, (void*)(&seqa), seqa_length};
  err |= hb_mc_device_dma_to_device(&device, &seqa_dma, 1);

  const int seqb_length = sizeof(char) * seqb.size();
  err  = hb_mc_device_malloc(&device, seqb_length, &seqb_dev);
  if (err != HB_MC_SUCCESS) {
    printf("failed to allocate memory on device for seqb.\n");
  }
  printf("start copying seqb to device\n");
  hb_mc_dma_htod_t seqb_dma = {seqb_dev, (void*)(&seqb), seqb_length};
  err |= hb_mc_device_dma_to_device(&device, &seqb_dma, 1);

  const int matrix_length = sizeof(int) * ((m+1)*(n+1));
  err  = hb_mc_device_malloc(&device, matrix_length, &matrix_dev);
  if (err != HB_MC_SUCCESS) {
    printf("failed to allocate memory on device for matrix.\n");
  }
  printf("start copying matrix to device\n");
  hb_mc_dma_htod_t matrix_dma = {matrix_dev, (void*)(hb_matrix), matrix_length};
  err |= hb_mc_device_dma_to_device(&device, &matrix_dma, 1);


  hb_mc_dimension_t grid_dim = { .x = 0, .y = 0};
  hb_mc_dimension_t tg_dim = { .x = 0, .y = 0 };
  hb_mc_dimension_t block_size = { .x = 0, .y = 0 };
  
  grid_dim = { .x = 1, .y = 1};
  tg_dim = { .x = 16, .y = 8 }; 
  /*---------------------------------------------------------------------
  * Prepare list of input arguments for kernel
  *---------------------------------------------------------------------*/

  uint32_t cuda_argv[8] = {seqa_dev, seqb_dev, matrix_dev, m, n, match, mismatch, gap};

  int cuda_argc = 8;

  printf("hb_mc_kernel_enqueue\n");
  err = hb_mc_kernel_enqueue(&device, grid_dim, tg_dim, "kernel_smith_waterman", cuda_argc, cuda_argv);

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

  printf("Copy device to Host.\n");
  err |= hb_mc_device_memcpy (&device, (void*)(hb_matrix), (void*)((intptr_t)matrix_dev),
                         matrix_size*sizeof(int), HB_MC_MEMCPY_TO_HOST);
  if (err != HB_MC_SUCCESS) {
    printf("ERROR: failed to copy matrix_dev to host\n");
  }  
  
  int error; 

  printf("check_dense.\n");
  error = check_result_correctness(cpu_matrix, hb_matrix, m, n);

  if (error != 0) {
    bsg_pr_test_err(BSG_RED("Mismatch. Error: %d\n"), error);
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

  free(cpu_matrix);
  free(hb_matrix);

  return HB_MC_SUCCESS;
}

declare_program_main("test_smith_waterman", test_smith_waterman);
