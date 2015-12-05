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


__global__ void cuda_test_kernel(int* num, int *num2){
    *num += *num2;
}


bool cuda_test() {
    int *dev_num, *dev_num2; // declare device int pointers
    int host_num, host_num2, host_outputNum; // declare host integers
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Allocate device memory
    cudaMalloc( &dev_num, sizeof(int) );
    cudaMalloc( &dev_num2, sizeof(int) );
    
    // Initialize host integers
    host_num = 3;
    host_num2 = 5;
    
    // Copy host integers to device
    cudaMemcpy( dev_num, &host_num, sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_num2, &host_num2, sizeof(int), cudaMemcpyHostToDevice );
    
    // call kernel
    cuda_test_kernel<<<1,1>>>(dev_num, dev_num2);
	
    // copy back (using cudaMemcpy) gpu_img_out to img_out
    cudaMemcpy( &host_outputNum, dev_num, sizeof(int), cudaMemcpyDeviceToHost );

    // free gpu memory
    cudaFree(dev_num);
    cudaFree(dev_num2);
    //printf("Processing time of test: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    int actualSum = host_num + host_num2;

    return (actualSum == host_outputNum);
}

// __global__
// void saxpy(int n, float a, float *x, float *y)
// {
//   int i = blockIdx.x*blockDim.x + threadIdx.x;
//   if (i < n) y[i] = a*x[i] + y[i];
// }
// 
// int main(void)
// {
//   int N = 1<<20;
//   float *x, *y, *d_x, *d_y;
//   x = (float*)malloc(N*sizeof(float));
//   y = (float*)malloc(N*sizeof(float));
// 
//   cudaMalloc(&d_x, N*sizeof(float)); 
//   cudaMalloc(&d_y, N*sizeof(float));
// 
//   for (int i = 0; i < N; i++) {
//     x[i] = 1.0f;
//     y[i] = 2.0f;
//   }
// 
//   cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
// 
//   // Perform SAXPY on 1M elements
//   saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
// 
//   cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
// 
//   float maxError = 0.0f;
//   for (int i = 0; i < N; i++)
//     maxError = max(maxError, abs(y[i]-4.0f));
//   printf("Max error: %fn", maxError);
// }