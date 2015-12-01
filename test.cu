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
#include <cuda.h>


__global__ void cuda_test_kernel(int* num, int num2){
    *num +=num2;
}


void cuda_test(){
    
    int * num;
    int init_num = 5;
    int init_num2 = 4;
    int output_num = 0;
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);


    // Allocate memory for num
    cudaMalloc( (void**) &num, sizeof(int) );
    
    // Copy init_num
    cudaMemcpy( num, &init_num, sizeof(int), cudaMemcpyHostToDevice );

    
    // call kernel
    cuda_test_kernel<<<1,1>>>(num, init_num2);

    // copy back (using cudaMemcpy) gpu_img_out to img_out
    cudaMemcpy( &output_num, num, sizeof(int), cudaMemcpyDeviceToHost );

    // free gpu memory
    cudaFree(num);
    printf("Processing time of test: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    printf("output_num: %i \n", output_num);
}

