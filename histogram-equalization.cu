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


__global__ void histogram_work()
void gpu_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    //int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;

    // Set up pointers for gpu device memory
    unsigned char * gpu_img_in, * gpu_img_out;
    int * gpu_hist_in, * gpu_lut;

    // Allocate memory for img_in, hist_in, lut, and img_out
    HANDLE_ERROR( cudaMalloc( (void**)&gpu_img_in, img_size * sizeof(unsigned char) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&gpu_img_out, img_size * sizeof(unsigned char) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&gpu_hist_in, (nbr_bin-1) * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&gpu_lut, (nbr_bin-1) * sizeof(int) ) );

    // Copy img_in and hist_in to gpu
    HANDLE_ERROR( cudaMemcpy( gpu_img_in, img_in, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( gpu_hist_in, hist_in, (nbr_bin-1) * sizeof(int), cudaMemcpyHostToDevice ) );

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

    // gpu version

    //call kernel
    // copy back (using cudaMemcpy) gpu_img_out to img_out
}

