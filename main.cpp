#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h> // CUDA Runtime
#include <helper_cuda.h> // Utility and system includes
#include <helper_functions.h> // helper for shared that are common to CUDA Samples
#include <helper_timer.h>
#include <getopt.h> // For reading command line input
#include "hist-equ.h" // contains cuda function prototypes

const int DEFAULT_THREADS_PER_BLOCK = 512;

// Function prototypes
void run_cpu_gray_test(PGM_IMG img_in);
void run_gpu_gray_test(PGM_IMG img_in, int blocksPerGrid, int threadsPerBlock);
void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in, int blocksPerGrid, int threadsPerBlock);


static struct option long_options[] =
{
    {"Input grayscale image", optional_argument, 0, 'g'},
    {"Input color image", optional_argument, 0, 'c'},
    {"Blocks per grid", optional_argument, 0, 'b'},
    {"Threads per block", optional_argument, 0, 't'},
    {0, 0, 0}
};

int main(int argc, char** argv) {
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;
    
    /// Set default values
    char *inputPGM = (char *)"in.pgm";
    char *inputPPM = (char *)"in.ppm";
    int blocksPerGrid = 0;
    int threadsPerBlock = 0;
    
    // Read input options
    while (true) {
        int option_index = 0;
        int c = getopt_long_only(argc, argv, "g:c:b:t:", long_options, &option_index);
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                break;
                
            case 'g':
                inputPGM = optarg;
                break;
                
            case 'c':
                inputPPM = optarg;
                break;
                
            case 'b':
                blocksPerGrid = atoi(optarg);
                break;
                
            case 't':
                threadsPerBlock = atoi(optarg);
                break;
                
            default:
                exit(1);
        }
    }
    
    printf("Input grayscale image:    %s\n", inputPGM);
    printf("Input color image:        %s\n", inputPPM);
    if (blocksPerGrid == 0) {
        printf("Blocks per grid:          default\n");
    } else {
        printf("Blocks per grid:          %d\n", blocksPerGrid);
    }
    if (threadsPerBlock == 0) {
        printf("Threads per block:        default\n\n");
    } else {
        printf("Threads per block:        %d\n\n", threadsPerBlock);
    }
    
    // Ensure cuda is setup and functioning correctly.
    // Also does first cudaMalloc, taking care of extra cost of first cudaMalloc
    assert(cuda_test());
    
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(inputPGM);
    run_cpu_gray_test(img_ibuf_g);
    run_gpu_gray_test(img_ibuf_g, blocksPerGrid, threadsPerBlock);
    free_pgm(img_ibuf_g);
    
    printf("\nRunning contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm(inputPPM);
    run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c, blocksPerGrid, threadsPerBlock);
    free_ppm(img_ibuf_c);
    
    return 0;
}



void run_gpu_color_test(PPM_IMG img_in, int blocksPerGrid, int threadsPerBlock) {
    printf("Starting GPU processing...\n");
    //TODO: run your GPU implementation here
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    bool CPU_timer = true;
    
    // Check for valid blocksPerGrid/threadsPerBlock
    if (threadsPerBlock == 0) {
        threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    }
    if (blocksPerGrid == 0) {
        blocksPerGrid = (img_in.h * img_in.w)/threadsPerBlock + 1;
    }
    
    if (CPU_timer) {
        StopWatchInterface *timer=NULL;
        
        // Perform HSL constrast enhancement
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
        img_obuf_hsl = gpu_contrast_enhancement_c_hsl(img_in, blocksPerGrid, threadsPerBlock);
        sdkStopTimer(&timer);
        printf("   HSL processing time (CPU timer): %f (ms)\n", sdkGetTimerValue(&timer));
        sdkDeleteTimer(&timer);
        write_ppm(img_obuf_hsl, "out_hsl_gpu.ppm");
        
        // Perform YUV constrast enhancement
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
        img_obuf_yuv = gpu_contrast_enhancement_c_yuv(img_in, blocksPerGrid, threadsPerBlock);
        sdkStopTimer(&timer);
        printf("   YUV processing time (CPU timer): %f (ms)\n", sdkGetTimerValue(&timer));
        sdkDeleteTimer(&timer);
        write_ppm(img_obuf_yuv, "out_yuv_gpu.ppm");
    } else {
        cudaEvent_t startHSL, stopHSL, startYUV, stopYUV;
        float hslExecutionMS, yuvExecutionMS;
        cudaEventCreate(&startHSL);
        cudaEventCreate(&stopHSL);
        cudaEventCreate(&startYUV);
        cudaEventCreate(&stopYUV);
        
        // Perform HSL constrast enhancement
        cudaEventRecord(startHSL);
        img_obuf_hsl = gpu_contrast_enhancement_c_hsl(img_in, blocksPerGrid, threadsPerBlock);
        cudaEventRecord(stopHSL);
        cudaEventSynchronize(stopHSL);
        cudaEventElapsedTime(&hslExecutionMS, startHSL, stopHSL);
        printf("   HSL processing time (GPU events): %f (ms)\n", hslExecutionMS);
        write_ppm(img_obuf_hsl, "out_hsl_gpu.ppm");
        
        // Perform YUV constrast enhancement
        cudaEventRecord(startYUV);
        img_obuf_yuv = gpu_contrast_enhancement_c_yuv(img_in, blocksPerGrid, threadsPerBlock);
        cudaEventRecord(stopYUV);
        cudaEventSynchronize(stopYUV);
        cudaEventElapsedTime(&yuvExecutionMS, startYUV, stopYUV);
        printf("   YUV processing time (GPU events): %f (ms)\n", yuvExecutionMS);
        write_ppm(img_obuf_yuv, "out_yuv_gpu.ppm");
    }
    
    
    //free_ppm(img_obuf_yuv);
    free_ppm(img_obuf_hsl);
}



void run_gpu_gray_test(PGM_IMG img_in, int blocksPerGrid, int threadsPerBlock) {
    printf("Starting GPU processing...\n");
    //TODO: run your GPU implementation here
    
    StopWatchInterface *timer = NULL;
    cudaEvent_t startEvent, stopEvent;
    float executionMS;
    PGM_IMG img_obuf;
    bool CPU_timer = true;
    
    // Check for valid blocksPerGrid/threadsPerBlock
    if (threadsPerBlock == 0) {
        threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    }
    if (blocksPerGrid == 0) {
        blocksPerGrid = (img_in.h * img_in.w)/threadsPerBlock + 1;
    }
    
    if (CPU_timer) {
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
        
        img_obuf = gpu_contrast_enhancement_g(img_in, blocksPerGrid, threadsPerBlock);
        
        sdkStopTimer(&timer);
        printf("   Processing time (CPU timer): %f (ms)\n", sdkGetTimerValue(&timer));
        sdkDeleteTimer(&timer);
    } else {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        
        cudaEventRecord(startEvent);
        
        img_obuf = gpu_contrast_enhancement_g(img_in, blocksPerGrid, threadsPerBlock);
        
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&executionMS, startEvent, stopEvent);
        
        printf("   Processing time (GPU events): %f (ms)\n", executionMS);
    }
    
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
    
    //free_ppm(img_obuf_yuv);
    free_ppm(img_obuf_hsl);
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
