#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h> // CUDA Runtime
#include <helper_cuda.h> // Utility and system includes
#include <helper_functions.h> // helper for shared that are common to CUDA Samples
#include <helper_timer.h>
#include "hist-equ.h" // contains cuda function prototypes

// Function prototypes
void run_cpu_gray_test(PGM_IMG img_in);
void run_gpu_gray_test(PGM_IMG img_in);
void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in);



int main() {
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;
    
    // Ensure cuda is setup and functioning correctly
    assert(cuda_test());
    
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm("in.pgm");
    run_cpu_gray_test(img_ibuf_g);
    run_gpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g);
    
    printf("\nRunning contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm("in.ppm");
    run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    return 0;
}



void run_gpu_color_test(PPM_IMG img_in) {
    printf("Starting GPU processing...\n");
    //TODO: run your GPU implementation here
    
    StopWatchInterface *timer=NULL;
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    
    // Perform HSL constrast enhancement
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_hsl = gpu_contrast_enhancement_c_hsl(img_in);
    sdkStopTimer(&timer);
    printf("   HSL processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_ppm(img_obuf_hsl, "out_hsl_gpu.ppm");
    
    // Perform HSL constrast enhancement
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    //img_obuf_yuv = gpu_contrast_enhancement_c_yuv(img_in);
    sdkStopTimer(&timer);
    printf("   YUV processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_ppm(img_obuf_yuv, "out_yuv_gpu.ppm");
    
    free_ppm(img_obuf_hsl);
    //free_ppm(img_obuf_yuv);
}



void run_gpu_gray_test(PGM_IMG img_in) {
    StopWatchInterface *timer = NULL;
    PGM_IMG img_obuf;
    
    printf("Starting GPU processing...\n");
    //TODO: run your GPU implementation here
    
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    img_obuf = gpu_contrast_enhancement_g(img_in);
    
    sdkStopTimer(&timer);
    printf("   Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_pgm(img_obuf, "out_bw_gpu.pgm");
    free_pgm(img_obuf);
}



void run_cpu_color_test(PPM_IMG img_in) {
    StopWatchInterface *timer=NULL;
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    
    printf("Starting CPU processing...\n");
    
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    sdkStopTimer(&timer);
    printf("   HSL processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_ppm(img_obuf_hsl, "out_hsl.ppm");
    
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    sdkStopTimer(&timer);
    printf("   YUV processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_ppm(img_obuf_yuv, "out_yuv.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}



void run_cpu_gray_test(PGM_IMG img_in) {
    StopWatchInterface *timer = NULL;
    PGM_IMG img_obuf;
    
    printf("Starting CPU processing...\n");
    
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    img_obuf = contrast_enhancement_g(img_in);
    
    sdkStopTimer(&timer);
    printf("   Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    write_pgm(img_obuf, "out_bw.pgm");
    free_pgm(img_obuf);
}



#pragma mark Grayscale image read/write/free functions

PGM_IMG read_pgm(const char * path) {
    FILE * in_file;
    char sbuf[256];
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL) {
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char) );
    
    
    fread(result.img,sizeof(unsigned char), result.w * result.h, in_file);
    fclose(in_file);
    
    return result;
}



void write_pgm(PGM_IMG img, const char * path) {
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}



void free_pgm(PGM_IMG img) {
    free(img.img);
}



#pragma mark Color image read/write/free functions

PPM_IMG read_ppm(const char * path) {
    FILE *in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max;
    in_file = fopen(path, "r");
    if (in_file == NULL) {
        printf("Input file not found!\n");
        exit(1);
    }
    
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);

    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    
    // Allocate memory for image
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         =          (char *)malloc(3 * result.w * result.h * sizeof(char));
    
    // Read image file into input bufer
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);
    
    // Separate RGB channels in input buffer and store in result.img_X
    for(int i = 0; i < result.w*result.h; i ++) {
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}



void write_ppm(PPM_IMG inputImage, const char * destinationPath) {
    FILE *outputFile;

    char *outputBuffer = (char *)malloc(3 * inputImage.w * inputImage.h * sizeof(char));
    
    // Copy input
    for(int i = 0; i < inputImage.w*inputImage.h; i ++){
        outputBuffer[3*i + 0] = inputImage.img_r[i];
        outputBuffer[3*i + 1] = inputImage.img_g[i];
        outputBuffer[3*i + 2] = inputImage.img_b[i];
    }
    
    outputFile = fopen(destinationPath, "wb");
    fprintf(outputFile, "P6\n");
    fprintf(outputFile, "%d %d\n255\n",inputImage.w, inputImage.h);
    fwrite(outputBuffer, sizeof(unsigned char), 3*inputImage.w*inputImage.h, outputFile);
    fclose(outputFile);
    free(outputBuffer);
}



void free_ppm(PPM_IMG img) {
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}