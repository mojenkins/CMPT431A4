#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>
// helper for shared that are common to CUDA Samples
#include <helper_functions.h>


__global__ void histogram_work(int d, int min, int nbr_bin, int* gpu_cdf, 
                        int* gpu_lut, unsigned char* gpu_img_in, unsigned char* gpu_img_out){
    if (blockIdx.x < 256){
        int lut = (int)(((float)gpu_cdf[blockIdx.x] - min)*255/d + 0.5);
        gpu_lut[blockIdx.x] = (lut <= 0)? 0 : lut;
    }

    gpu_img_out[blockIdx.x] = (gpu_lut[gpu_img_in[blockIdx.x]] > 255) ? 255 : gpu_lut[gpu_img_in[blockIdx.x]];
}


void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin){
    //int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, min, d;
    int cdf[256];

    /* Construct the LUT by calculating the CDF */
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;

    // Set up pointers for gpu device memory
    unsigned char * gpu_img_in, * gpu_img_out;
    int  * gpu_lut, * gpu_cdf;

    // Allocate memory for img_in, hist_in, lut, and img_out
    cudaMalloc( (void**)&gpu_img_in, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_lut, (nbr_bin) * sizeof(int) );
    cudaMalloc( (void**)&gpu_cdf, sizeof(int) );

    // Calculate cdf sequentially
    cdf[0] = hist_in[0];
    for(i = 1; i < nbr_bin; i ++){
        cdf[i] = cdf[i-1] + hist_in[i];
    }

    // Copy img_in and cdf to gpu
    cudaMemcpy( gpu_img_in, img_in, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_cdf, cdf, (nbr_bin) * sizeof(int), cudaMemcpyHostToDevice );

    
    // sequential version
    // for(i = 0; i < nbr_bin; i ++){
    //     cdf += hist_in[i];
    //     //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
    //     lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    //     if(lut[i] < 0){
    //         lut[i] = 0;
    //     }
        
        
    // }
    
    // /* Get the result image */
    // for(i = 0; i < img_size; i ++){
    //     if(lut[img_in[i]] > 255){
    //         img_out[i] = 255;
    //     }
    //     else{
    //         img_out[i] = (unsigned char)lut[img_in[i]];
    //     }
        
    // }

    // GPU version
    // call kernel
    histogram_work<<<img_size,1>>>(d, min, nbr_bin, gpu_cdf, gpu_lut, gpu_img_in, gpu_img_out);

    // copy back (using cudaMemcpy) gpu_img_out to img_out
    cudaMemcpy( img_out, gpu_img_out, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );

    // free gpu memory
    cudaFree(gpu_img_out);
    cudaFree(gpu_img_in);
    cudaFree(gpu_lut);
    cudaFree(gpu_cdf);
}

