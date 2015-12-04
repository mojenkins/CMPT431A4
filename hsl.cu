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

const int imageBitDepth = 8*8; //*** Hard coded. Only 8-bit images supported

__global__ void rgb2hsl_kernel(int img_size, PPM_IMG* gpu_img_in, HSL_IMG* gpu_img_out) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < img_size){
        float H, S, L;

        // Convert RGB from [0,255] to [0,1]
        float var_r = ( (float)gpu_img_in->img_r[index]/(imageBitDepth-1) );
        float var_g = ( (float)gpu_img_in->img_g[index]/(imageBitDepth-1) );
        float var_b = ( (float)gpu_img_in->img_b[index]/(imageBitDepth-1) );
        
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
        
        // Ensure valid H value
        if (H < 0) {
            H += 1;
        }
        if (H > 1) {
            H -= 1;
        }
        
        // Save HSL values to output image
        gpu_img_out->h[index] = H;
        gpu_img_out->s[index] = S;
        gpu_img_out->l[index] = (unsigned char)(L*255);
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
    
    // Allocate pointers to image objects to be processed on the GPU
    PPM_IMG *dev_img_in_rgb = (PPM_IMG *)malloc(sizeof(PPM_IMG));
    HSL_IMG *dev_img_out_hsl = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    
    // Allocate memory for RGB channels of dev_img_in_rgb and HSL channels of dev_img_out_hsl
    cudaMalloc( &(dev_img_in_rgb->img_r), img_size * sizeof(unsigned char) );
    cudaMalloc( &(dev_img_in_rgb->img_g), img_size * sizeof(unsigned char) );
    cudaMalloc( &(dev_img_in_rgb->img_b), img_size * sizeof(unsigned char) );
    cudaMalloc( &(dev_img_out_hsl->h), img_size * sizeof(float) );
    cudaMalloc( &(dev_img_out_hsl->s), img_size * sizeof(float) );
    cudaMalloc( &(dev_img_out_hsl->l), img_size * sizeof(unsigned char) );
    
    // Copy img_in to dev_img_in_rgb
    cudaMemcpy( dev_img_in_rgb->img_r, img_in.img_r, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_img_in_rgb->img_g, img_in.img_g, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_img_in_rgb->img_b, img_in.img_b, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    
    // Execute kernel
    rgb2hsl_kernel<<<img_size/512+1,512>>>(img_size, dev_img_in_rgb, dev_img_out_hsl);
    
    // Copy device HSL data to img_out
    cudaMemcpy( img_out.h, dev_img_out_hsl->h, img_size * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.s, dev_img_out_hsl->s, img_size * sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.l, dev_img_out_hsl->l, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    
    // free GPU memory
    cudaFree(dev_img_in_rgb->img_r);
    cudaFree(dev_img_in_rgb->img_g);
    cudaFree(dev_img_in_rgb->img_b);
    cudaFree(dev_img_out_hsl->h);
    cudaFree(dev_img_out_hsl->s);
    cudaFree(dev_img_out_hsl->l);
    
    return img_out;
    
}