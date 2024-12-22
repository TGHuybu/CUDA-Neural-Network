#include "utils_device.h"


void device_info() {
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}

__global__ void _matmul_GPU(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    	if (row < m && col < k)
    	{
        	float value = 0;
        	for (int i = 0; i < n; i++) 
        	{
            		value += A[row * n + i] * B[i * k + col];
        	}
        	C[row * k + col] = value; 
    	}
}

// void matMul(float* A, float* B, float* C, int m, int n, int k, dim3 blockSize = dim3(1)){

//         float* d_A, * d_B, * d_C;
// 	CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
// 	CHECK(cudaMalloc((void**)&d_B, n * k * sizeof(float)));
// 	CHECK(cudaMalloc((void**)&d_C, m * k * sizeof(float)));

//         // TODO: Copy data to device memories
//         CHECK(cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));
// 	CHECK(cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice));
//         dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
//  // TODO: Compute gridSize
        
		
// 		matMulkernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

//         // TODO: Copy result from device memory
// 	CHECK(cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));
//         // TODO: Free device memories
// 	CHECK(cudaFree(d_A));
// 	CHECK(cudaFree(d_B));
// 	CHECK(cudaFree(d_C));
	
// 		printf("Grid size: %d * %d, block size: %d * %d\n", 
// 			gridSize.x,gridSize.y, blockSize.x,blockSize.y);

// }

__global__ void _ReLU_GPU(float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Z[idx] = fmaxf(0.0f, Z[idx]);
    }
}

// __global__ void softmax(float *input, float *output, int batch_size, int output_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int batch_idx = idx / output_size;
//     int output_idx = idx % output_size;

//     if (batch_idx >= batch_size) return;

//     // Shared memory for stable softmax
//     __shared__ float max_val[1024];
//     __shared__ float sum[1024];

//     // Find maximum value in the row
//     float local_max = -FLT_MAX;
//     for (int i = 0; i < output_size; ++i) {
//         local_max = max(local_max, input[batch_idx * output_size + i]);
//     }
//     max_val[threadIdx.x] = local_max;
//     __syncthreads();

//     // Calculate exp(input - max)
//     float exp_sum = 0.0f;
//     for (int i = 0; i < output_size; ++i) {
//         exp_sum += exp(input[batch_idx * output_size + i] - max_val[threadIdx.x]);
//     }
//     sum[threadIdx.x] = exp_sum;
//     __syncthreads();

//     // Normalize
//     output[idx] = exp(input[idx] - max_val[threadIdx.x]) / sum[threadIdx.x];
// }

__global__ void _softmax_GPU(float *input, float *output, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / output_size;
    int output_idx = idx % output_size;

    if (batch_idx >= batch_size) return;

    // Find maximum value in the row
    float local_max = -1;   // temp in place of old const? 
    for (int i = 0; i < output_size; ++i) {
        local_max = max(local_max, input[batch_idx * output_size + i]);
    }

    // Calculate exp(input - max) and sum_exp for normalization
    float exp_sum = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        exp_sum += exp(input[batch_idx * output_size + i] - local_max);
    }

    // Normalize and output the result
    output[idx] = exp(input[batch_idx * output_size + output_idx] - local_max) / exp_sum;
}

