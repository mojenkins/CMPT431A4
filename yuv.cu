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

__global__ void rgb2yuv_work(int img_size, unsigned char* gpu_img_in_r, unsigned char* gpu_img_in_g, unsigned char* gpu_img_in_b,
				 unsigned char* gpu_img_out_y, unsigned char* gpu_img_out_u, unsigned char* gpu_img_out_v){
	unsigned char r, g, b;

	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < img_size){
		r = gpu_img_in_r[index];
		g = gpu_img_in_g[index];
		b = gpu_img_in_b[index];

	    gpu_img_out_y[index] = (unsigned char) (0.299*r + 0.587*g +  0.114*b);
	    gpu_img_out_u[index] = (unsigned char) (-0.169*r - 0.331*g +  0.499*b + 128);
	    gpu_img_out_v[index] = (unsigned char) (0.499*r - 0.418*g - 0.0813*b + 128);
	}
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG gpu_rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int img_size = img_in.w*img_in.h;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc( sizeof(unsigned char)*img_size );
    img_out.img_u = (unsigned char *)malloc( sizeof(unsigned char)*img_size );
    img_out.img_v = (unsigned char *)malloc( sizeof(unsigned char)*img_size );

    // Sequential version
    // for(i = 0; i < img_out.w*img_out.h; i ++){
    //     r = img_in.img_r[i];
    //     g = img_in.img_g[i];
    //     b = img_in.img_b[i];
        
    //     y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
    //     cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
    //     cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
    //     img_out.img_y[i] = y;
    //     img_out.img_u[i] = cb;
    //     img_out.img_v[i] = cr;
    // }

    //GPU version
    // Set up pointers for gpu device memory
    unsigned char * gpu_img_in_r, * gpu_img_in_g, * gpu_img_in_b, * gpu_img_out_y, * gpu_img_out_u, * gpu_img_out_v;

    // Allocate memory on GPU
    cudaMalloc( (void**)&gpu_img_in_r, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_in_g, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_in_b, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_y, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_u, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_v, img_size * sizeof(unsigned char) );

    // Copy input image to gpu
    cudaMemcpy( gpu_img_in_r, img_in.img_r, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_img_in_g, img_in.img_g, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_img_in_b, img_in.img_b, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );

    // call kernel
    rgb2yuv_work<<<img_size/512+1,512>>>(img_size, gpu_img_in_r, gpu_img_in_g, gpu_img_in_b, gpu_img_out_y, gpu_img_out_u, gpu_img_out_v);

    // Copy resultant image from gpu
    cudaMemcpy( img_out.img_y, gpu_img_out_y, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.img_u, gpu_img_out_u, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.img_v, gpu_img_out_v, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );

    //free gpu memory
    cudaFree(gpu_img_in_r);
    cudaFree(gpu_img_in_g);
    cudaFree(gpu_img_in_b);
    cudaFree(gpu_img_out_y);
    cudaFree(gpu_img_out_u);
    cudaFree(gpu_img_out_v);

    return img_out;
}

__global__ void yuv2rgb_work(int img_size, unsigned char* gpu_img_in_y, unsigned char* gpu_img_in_u, unsigned char* gpu_img_in_v,
				 unsigned char* gpu_img_out_r, unsigned char* gpu_img_out_g, unsigned char* gpu_img_out_b){
	int rt,gt,bt;
	int rt2, gt2, bt2;

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < img_size){

	    rt  = (int) (gpu_img_in_y[index] + 1.402*(gpu_img_in_v[index]-128));
	    gt  = (int) (gpu_img_in_y[index] - 0.344*(gpu_img_in_u[index]-128) - 0.714*(gpu_img_in_v[index]-128));
	    bt  = (int) gpu_img_in_y[index] + 1.772*(gpu_img_in_u[index]-128);

	    rt2 = (rt > 255) ? 255 : rt;
	    gt2 = (gt > 255) ? 255 : gt;
	    bt2 = (bt > 255) ? 255 : bt;

	    gpu_img_out_r[index] = (rt2 < 0) ? 0 : rt2;
	    gpu_img_out_b[index] = (bt2 < 0) ? 0 : bt2;
	    gpu_img_out_g[index] = (gt2 < 0) ? 0 : gt2;
	}
}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG gpu_yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    
    int img_size = img_in.w*img_in.h;
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_size);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_size);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_size);

    //Sequential version
    // for(i = 0; i < img_out.w*img_out.h; i ++){
    //     y  = (int)img_in.img_y[i];
    //     cb = (int)img_in.img_u[i] - 128;
    //     cr = (int)img_in.img_v[i] - 128;
        
    //     rt  = (int)( y + 1.402*cr);
    //     gt  = (int)( y - 0.344*cb - 0.714*cr);
    //     bt  = (int)( y + 1.772*cb);

    //     img_out.img_r[i] = clip_rgb(rt);
    //     img_out.img_g[i] = clip_rgb(gt);
    //     img_out.img_b[i] = clip_rgb(bt);
    // }

    //GPU version
    // Set up pointers for gpu device memory
    unsigned char * gpu_img_in_y, * gpu_img_in_u, * gpu_img_in_v, * gpu_img_out_r, * gpu_img_out_g, * gpu_img_out_b;

    // Allocate memory on GPU
    cudaMalloc( (void**)&gpu_img_in_y, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_in_u, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_in_v, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_r, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_g, img_size * sizeof(unsigned char) );
    cudaMalloc( (void**)&gpu_img_out_b, img_size * sizeof(unsigned char) );

    // Copy input image to gpu
    cudaMemcpy( gpu_img_in_y, img_in.img_y, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_img_in_u, img_in.img_u, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_img_in_v, img_in.img_v, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice );

    // call kernel
    yuv2rgb_work<<<img_size/512+1,512>>>(img_size, gpu_img_in_y, gpu_img_in_u, gpu_img_in_v, gpu_img_out_r, gpu_img_out_g, gpu_img_out_b);

    // Copy resultant image from gpu
    cudaMemcpy( img_out.img_r, gpu_img_out_r, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.img_g, gpu_img_out_g, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    cudaMemcpy( img_out.img_b, gpu_img_out_b, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );

    //free gpu memory
    cudaFree(gpu_img_in_y);
    cudaFree(gpu_img_in_u);
    cudaFree(gpu_img_in_v);
    cudaFree(gpu_img_out_r);
    cudaFree(gpu_img_out_g);
    cudaFree(gpu_img_out_b);

    return img_out;
}
