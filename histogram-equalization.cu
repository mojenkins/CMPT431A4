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


#pragma mark GPU Histogram Equalization

__global__ void histogram_work(int img_size, int* gpu_lut, unsigned char* gpu_img_in, unsigned char* gpu_img_out){
    int i = blockIdx.x + threadIdx.x;
    if (i < img_size){
        if (gpu_lut[gpu_img_in[i]] > 255) {
            gpu_img_out[i] = 255;
        } else {
            gpu_img_out[i] = (unsigned char)gpu_lut[gpu_img_in[i]];
        }
    }
}


void gpu_histogram_equalization(unsigned char * img_out,
                                unsigned char * img_in,
                                int * hist_in,
                                int img_size,
                                int nbr_bin) {
    
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, min, d, cdf;
    
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    while(min == 0){
        min = hist_in[i++];
    }
    
    d = img_size - min;
    
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }
    
    // Set up pointers for gpu device memory
    unsigned char * gpu_img_in, * gpu_img_out;
    int  * gpu_lut; //, * gpu_cdf;
    
    // Allocate memory for img_in, hist_in, lut, and img_out
    cudaMalloc( (void**)&gpu_img_in, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_lut, (nbr_bin) * sizeof(int) );
    
    // Copy img_in and cdf to gpu
    cudaMemcpy( gpu_img_in, img_in, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_lut, lut, (nbr_bin) * sizeof(int), cudaMemcpyHostToDevice );
    
    // GPU version
    // call kernel
    histogram_work<<<img_size/512+1,512>>>(img_size, gpu_lut, gpu_img_in, gpu_img_out);
    
    // copy back (using cudaMemcpy) gpu_img_out to img_out
    cudaMemcpy( img_out, gpu_img_out, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    
    // free gpu memory
    cudaFree(gpu_img_out);
    cudaFree(gpu_img_in);
    cudaFree(gpu_lut);
}


#pragma mark GPU RGB to HSL Conversion

__global__ void rgb2hsl_kernel(int img_size, PPM_IMG* gpu_img_in, HSL_IMG* gpu_img_out) {
    if (blockIdx.x + threadIdx.x < img_size){
        //gpu_img_out[blockIdx.x + threadIdx.x] = (gpu_lut[gpu_img_in[blockIdx.x + threadIdx.x]] > 255) ? 255 : gpu_lut[gpu_img_in[blockIdx.x + threadIdx.x]];
        
        float H, S, L;

        // Convert RGB from [0,255] to [0,1]
        float var_r = ( (float)gpu_img_in->img_r[blockIdx.x + threadIdx.x]/(imageBitDepth-1) );
        float var_g = ( (float)gpu_img_in->img_g[blockIdx.x + threadIdx.x]/(imageBitDepth-1) );
        float var_b = ( (float)gpu_img_in->img_b[blockIdx.x + threadIdx.x]/(imageBitDepth-1) );
        
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
        gpu_img_out->h[blockIdx.x + threadIdx.x] = H;
        gpu_img_out->s[blockIdx.x + threadIdx.x] = S;
        gpu_img_out->l[blockIdx.x + threadIdx.x] = (unsigned char)(L*255);
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