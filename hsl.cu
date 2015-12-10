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

const int imageDepth = 256; //*** Hard coded. Only 8-bit images supported 

__global__ void rgb2hsl_kernel(int img_size, unsigned char *gpu_img_in_r, unsigned char *gpu_img_in_g, unsigned char *gpu_img_in_b, float *gpu_img_out_h, float *gpu_img_out_s, unsigned char *gpu_img_out_l) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < img_size){
        float H, S, L;
        
        // Convert RGB from [0,255] to [0,1]
        float var_r = ( (float)gpu_img_in_r[index]/(imageDepth-1) );
        float var_g = ( (float)gpu_img_in_g[index]/(imageDepth-1) );
        float var_b = ( (float)gpu_img_in_b[index]/(imageDepth-1) );
        
        // Find min and max values
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        // Calculate L
        L = (var_max + var_min) / 2;
        
        // Calculate S
        if (del_max == 0) {
            S = 0;
        } else if (L < 0.5) {
            S = del_max/(var_max+var_min);
        } else {
            S = del_max/(2-var_max-var_min );
        }
        
        // Calculate H
        float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
        float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
        float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
        
        if (del_max == 0) {
            H = 0;
        } else if (var_r == var_max) {
            H = del_b - del_g;
        } else if (var_g == var_max) {
            H = (1.0/3.0) + del_r - del_b;
        } else {
            H = (2.0/3.0) + del_g - del_r;
        }

        
        if (H < 0) {
            H += 1;
        } else if (H > 1) {
            H -= 1;
        }

        // Save HSL values to output image
        gpu_img_out_h[index] = H;
        gpu_img_out_s[index] = S;
        gpu_img_out_l[index] = (unsigned char)(L*255);
    }
}


//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG gpu_rgb2hsl(PPM_IMG img_in) {
    int img_size = img_in.w * img_in.h;
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_size * sizeof(float));
    img_out.s = (float *)malloc(img_size * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    
    // Set up pointers for gpu device memory
    unsigned char *gpu_img_in_r, *gpu_img_in_g, *gpu_img_in_b, *gpu_img_out_l;
    float *gpu_img_out_h, *gpu_img_out_s;
    
    // Allocate memory on GPU
    cudaMalloc( (void**)&gpu_img_in_r, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_in_g, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_in_b, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_h, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_s, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_l, img_size * sizeof(unsigned char) );
    
    // Copy input image to gpu
    cudaMemcpy( gpu_img_in_r, img_in.img_r, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_img_in_g, img_in.img_g, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_img_in_b, img_in.img_b, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    
    // call kernel
    rgb2hsl_kernel<<<img_size/512+1,512>>>(img_size, gpu_img_in_r, gpu_img_in_g, gpu_img_in_b, gpu_img_out_h, gpu_img_out_s, gpu_img_out_l);
    
    // Copy resultant image from gpu
    cudaMemcpy( img_out.h, gpu_img_out_h, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.s, gpu_img_out_s, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.l, gpu_img_out_l, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    
    //free gpu memory
    cudaFree(gpu_img_in_r);
    cudaFree(gpu_img_in_g);
    cudaFree(gpu_img_in_b);
    cudaFree(gpu_img_out_h);
    cudaFree(gpu_img_out_s);
    cudaFree(gpu_img_out_l);
    
    return img_out;
    
}