// Copyright (c) 2019, University of Washington All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
// 
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// 
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/******************************************************************************/
/* A[N] * B[N] --> C[N]                                                       */
/* Runs the floating point vector multiplication on one 2x2 tile group        */
/* Grid dimensions are prefixed at 1x1. --> block_size_x is set to N.         */
/* This tests uses the software/spmd/bsg_cuda_lite_runtime/float_vec_mul/     */
/* manycore binary in the BSG Manycore repository.                            */
/******************************************************************************/


#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>

#include <vector>
#include "dirty_zipfian_int_distribution.h"

#define ALLOC_NAME "default_allocator"

typedef struct {
    int** paths;
    int* path_idx;
    int path_cnt;
} graph_t;

void host_float_vec_mul (int *path_idx, int ***paths) { 
        return;
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

int kernel_float_vec_mul (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA Floating Point Vector Multiplciation "
                         "Kernel on a 1x1 grid of 2x2 tile group.\n\n");

        srand(time(NULL)); 


        /**********************************************************************/
        /* Define path to binary.                                             */
        /* Initialize device, load binary and unfreeze tiles.                 */
        /**********************************************************************/
        hb_mc_device_t device;
        rc = hb_mc_device_init(&device, test_name, 0);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize device.\n");
                return rc;
        }


        rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize program.\n");
                return rc;
        }


        /**********************************************************************/
        /* Allocate memory on the device for A, B and C.                      */
        /**********************************************************************/
        FILE *file;
        char filename[] = "mod_psgd.mtx";
        file = fopen(filename, "r");
        if (file == NULL) {
                printf("Failed to open the file.\n");
                return 1; 
        }
        int nodes, path_cnt, path_len, *path_idx, **paths;
        if (fscanf(file, "%d", &nodes) != EOF) 
        {
                if (fscanf(file, "%d", &path_cnt) != EOF) 
                {
                        paths = (int **) malloc(path_cnt * sizeof(int *));
                        path_idx = (int *) malloc(path_cnt * sizeof(int));
                        for(int i = 0 ; i < path_cnt ; i++)
                        {
                                if(fscanf(file, "%d", &path_len) != EOF)
                                {
                                        path_idx[i] = path_len;
                                        paths[i] = (int *) malloc(sizeof(int) * path_len * 2);
                                        for(int j = 0 ; j < path_len ; j++)
                                        {
                                                fscanf(file, "%d", &paths[i][j*2+0]);
                                                fscanf(file, "%d", &paths[i][j*2+1]);
                                        }
                                }
                        }        
                }
        }
        fclose(file);

        hb_mc_eva_t path_idx_device, paths_device;

        //Copy Over the Path Index
        rc = hb_mc_device_malloc(&device, path_cnt * sizeof(int), &path_idx_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        void *dst = (void *) ((intptr_t) path_idx_device);
        void *src = (void *) &path_idx[0];
        rc = hb_mc_device_memcpy (&device, dst, src, path_cnt * sizeof(int), HB_MC_MEMCPY_TO_DEVICE);     
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        //Copy Over the Paths

        rc = hb_mc_device_malloc(&device, path_cnt * sizeof(hb_mc_eva_t), &paths_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        //hb_mc_eva_t * device_path_ptrs = (hb_mc_eva_t * ) malloc(sizeof(hb_mc_eva_t)*path_cnt);
        for(int i = 0 ; i  <  path_cnt ; i++)
        {
                hb_mc_eva_t path_i;
                rc = hb_mc_device_malloc(&device, path_idx[i] * sizeof(int) * 2, &path_i);
                if (rc != HB_MC_SUCCESS) 
                { 
                        bsg_pr_err("failed to allocate memory on device.\n");
                        return rc;
                }
                void *dst = (void *) ((intptr_t) path_i);
                void *src = (void *) &paths[i][0];
                rc = hb_mc_device_memcpy (&device, dst, src, path_idx[i] * sizeof(int) * 2, HB_MC_MEMCPY_TO_DEVICE);     
                if (rc != HB_MC_SUCCESS) 
                { 
                        bsg_pr_err("failed to copy memory to device.\n");
                        return rc;
                }
                rc = hb_mc_device_memcpy (&device, (void*)paths_device + i*sizeof(hb_mc_eva_t), &path_i, sizeof(hb_mc_eva_t), HB_MC_MEMCPY_TO_DEVICE);     
                if (rc != HB_MC_SUCCESS) 
                { 
                        bsg_pr_err("failed to copy memory to device.\n");
                        return rc;
                }
        }

        /**********************************************************************/
        /* Allocate memory on the host for X & Y                              */
        /* and initialize with random values.                                 */
        /**********************************************************************/
        uint32_t N = nodes+1;



        hb_mc_eva_t Eta_device; 
        double eta_max = 17000*17000;
        double w_min = (double) 1.0 / (double) (eta_max);
        double w_max = 1.0;
        double iter_max = 1; // CHANGE THIS TO CHANGE THE OUTER ITERATIONS
        uint64_t iter_with_max_learning_rate = 0;
        double eps = 0.1;
        double* etas = path_linear_sgd_layout_schedule (w_min, w_max, iter_max-1,
                                                                        iter_with_max_learning_rate,
                                                                        eps).data();

        rc = hb_mc_device_malloc(&device,  sizeof(double) * iter_max, &Eta_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        dst = (void *) ( Eta_device);
        src = (void *)  &etas[0];
        rc = hb_mc_device_memcpy (&device, dst, src, sizeof(double) * iter_max, HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        

        hb_mc_eva_t Zeta_device;
        int space = 0;
        for(int i = 0; i < path_cnt ; i++) space = std::max(space, path_idx[i]);
        int space_max = 1000;
        int space_quantization_step = 100;
        std::vector<double> zetas((space <= space_max ? space : space_max + (space - space_max) / space_quantization_step + 1)+1);
        double zeta_tmp = 0.0;
        double theta =  0.99;
        for (int i = 1; i < space + 1; i++) {
                zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / i, theta);
                if (i <= space_max) {
                        zetas[i] = zeta_tmp;
                }
                if (i >= space_max && (i - space_max) % space_quantization_step == 0) {
                        zetas[space_max + 1 + (i - space_max) / space_quantization_step] = zeta_tmp;
                }
        }
        rc = hb_mc_device_malloc(&device,  sizeof(double) * space + 1, &Zeta_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        dst = (void *) (Zeta_device);
        src = (void *)  &zetas[0];
        rc = hb_mc_device_memcpy (&device, dst, src, sizeof(double) * iter_max, HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        
        float X_host[N];
        float Y_host[N];
        float X_host_2[N];
        float Y_host_2[N];
        for(auto  i = 0; i < N; i++)
        {
                X_host[i] = rand()/10000.0;
                Y_host[i] = rand()/10000.0;

        }

        hb_mc_eva_t X_device; 
        rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &X_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        dst = (void *) (X_device);
        src = (void *)  &X_host[0];
        rc = hb_mc_device_memcpy (&device, dst, src, sizeof(float) * N, HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        hb_mc_eva_t Y_device; 
        rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &Y_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        dst = (void *) (Y_device);
        src = (void *)  &Y_host[0];
        rc = hb_mc_device_memcpy (&device, dst, src, sizeof(float) * N, HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        /**********************************************************************/
        /* Define block_size_x/y: amount of work for each tile group          */
        /* Define tg_dim_x/y: number of tiles in each tile group              */
        /* Calculate grid_dim_x/y: number of                                  */
        /* tile groups needed based on block_size_x/y                         */
        /**********************************************************************/

        hb_mc_dimension_t tile_group_dim = { .x = 1, .y = 1}; 

        hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 

        uint32_t cuda_argv[9] = {path_cnt,path_idx_device, paths_device, X_device, Y_device, iter_max, Eta_device, Zeta_device,nodes};

        
        /**********************************************************************/
        /* Enquque grid of tile groups, pass in grid and tile group dimensions*/
        /* kernel name, number and list of input arguments                    */
        /**********************************************************************/
        
        rc = hb_mc_kernel_enqueue (&device, grid_dim, tile_group_dim, "kernel_float_vec_mul", 9, cuda_argv);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize grid.\n");
                return rc;
        }


        /**********************************************************************/
        /* Launch and execute all tile groups on device and wait for finish.  */ 
        /**********************************************************************/
        rc = hb_mc_device_tile_groups_execute(&device);
        printf("\n\n\n Status: %d \n\n\n");
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to execute tile groups.\n");
                return rc;
        }
        src= (void *) ((intptr_t) path_idx_device);
        dst = (void *) &path_idx[0];
        rc = hb_mc_device_memcpy (&device, dst, src, path_cnt * sizeof(int), HB_MC_MEMCPY_TO_HOST);     
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        
        src = (void *) ((intptr_t) X_device);
        dst = (void *) &X_host_2[0];
        rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(float), HB_MC_MEMCPY_TO_HOST);     
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        src = (void *) ((intptr_t) Y_device);
        dst = (void *) &Y_host_2[0];
        rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(float), HB_MC_MEMCPY_TO_HOST);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        printf("------------output------------\n");
        for(int i = 0 ; i  < N ; i++)
        {   
                printf("%f %f\n", X_host_2[i], Y_host_2[i]);
        }
        printf("------------------------------\n");
        // /**********************************************************************/
        // /* Freeze the tiles and memory manager cleanup.                       */
        /**********************************************************************/
        rc = hb_mc_device_finish(&device); 
        if (rc != HB_MC_SUCCESS) 
        { 
                bsg_pr_err("failed to de-initialize device.\n");
                return rc;
        }
        return HB_MC_SUCCESS;
}

declare_program_main("test_float_vec_mul", kernel_float_vec_mul);
