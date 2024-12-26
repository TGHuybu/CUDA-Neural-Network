#ifndef UTILS_DEVICE_H
#define UTILS_DEVICE_H

#include "libs.h"

#define TILE_WIDTH 32

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    }\
}

struct GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() {
		cudaEventRecord(start,0);
		cudaEventSynchronize(start);
	}

	void Stop() {
		cudaEventRecord(stop, 0);
	}

	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

__global__ void relu_derivative(const float* , float* , int );

__global__ void scalar_div(float* , int , float );

__global__ void _transpose_GPU(float*, float*, int, int);

__global__ void _add_GPU(float*, float*, float*, int, float);

__global__ void _ewmul_GPU(float*, float*, float*, int);

__global__ void _matmul_GPU(float*, float*, float*, int, int, int);

__global__ void _tiled_matmul_GPU(float*, float*, float*, int, int, int);

__global__ void _sum_GPU(float*, float*, int);

__global__ void _ReLU_GPU(float*, int);

__global__ void _softmax_GPU(float *, float *, int , int ) ;

vector<float*> _forward_GPU(float*, vector<vector<float>>, int, int, 
                        	int, int, bool=true);

vector<float*> _backward_GPU(vector<float*> , vector<vector<float>> ,
								vector<float> , int , int ,
								int , int );

#endif