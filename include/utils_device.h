#ifndef UTILS_DEVICE_H
#define UTILS_DEVICE_H

#include "libs.h"


#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    }\
}


void device_info();

__global__ void _matmul_GPU(float*, float*, float*, int, int, int);

__global__ void _ReLU_GPU(float*, int);

__global__ void _softmax_GPU(float *, float *, int , int ) ;

#endif