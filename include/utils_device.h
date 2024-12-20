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
__global__ void matMul(float*, float*, float*, int, int, int);
__global__ void ReLU(float*, int);
__global__ void softmax(float *, float *, int , int ) ;
