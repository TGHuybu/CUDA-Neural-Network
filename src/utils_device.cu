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


__global__ void _transpose_GPU(float* A, float* A_T, int n_rows, int n_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_rows && j < n_cols) 
        A_T[n_rows * j + i] = A[n_cols * i + j];
}


__global__ void _add_CPU(float* A, float* B, float* C, int n, float sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + sign * B[idx]; 
}


__global__ void _ewmul_GPU(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] * B[idx]; 
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


__global__ void _tiled_matmul_GPU(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    float value = 0;

    // Shared memory cho A và B
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    for (int tile = 0; tile < (n + TILE_WIDTH - 1) / TILE_WIDTH; tile++)
    {
        if (row < m && (tile * TILE_WIDTH + threadIdx.x) < n)
            s_A[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_WIDTH + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0;

        if ((tile * TILE_WIDTH + threadIdx.y) < n && col < k)
            s_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_WIDTH + threadIdx.y) * k + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads(); 

        for (int i = 0; i < TILE_WIDTH; i++) {
            value += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads(); 
    }

    // Ghi kết quả vào ma trận C
    if (row < m && col < k) {
        C[row * k + col] = value;
    }
    	
}


__global__ void _sum_GPU(float* in, float* out, int n) {
    int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
    int i = numElemsBeforeBlk + threadIdx.x;

    for (int stride = blockDim.x; stride > 0; stride /= 2) {
        
        if (threadIdx.x < stride) {
            if (i < n && i + stride < n)
                in[i] += in[i + stride];
        }

        __syncthreads(); // Synchronize within each block
    }
    
    if (threadIdx.x == 0)
        atomicAdd(out, in[numElemsBeforeBlk]);
}


__global__ void _ReLU_GPU(float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Z[idx] = fmaxf(0.0f, Z[idx]);
    }
}


__global__ void _softmax_GPU(float *input, float *output, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / output_size;
    int output_idx = idx % output_size;

    if (batch_idx >= batch_size) return;

    // Find maximum value in the row
    float local_max = -1;
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


vector<float*> _fw_GPU(vector<float> X, vector<vector<float>> Ws, int n_samples, int n_features, 
                        int hidden_size, int out_size) {

    vector<float*> outs;
    outs.push_back(X.data());

    GpuTimer timer;
    float time;

    for (int i = 0; i < Ws.size(); i++) {
        if (i != 0) n_features = hidden_size;
        if (i == Ws.size() - 1) hidden_size = out_size;

        int n_input_elements = (n_samples * n_features);
        int n_output_elements = (n_samples * hidden_size);

        timer.Start();
 
        vector<float> W = Ws[i];
        float *X = outs[i];
        float *out;
        CHECK(cudaMallocHost(&out, n_output_elements * sizeof(float)));

        // Allocate memory on device
        float *d_X, *d_W, *d_out;
        CHECK(cudaMalloc(&d_X, n_input_elements * sizeof(float)));
        CHECK(cudaMalloc(&d_W, W.size() * sizeof(float)));
        CHECK(cudaMalloc(&d_out, n_output_elements * sizeof(float)));

        // Copy memory: host-to-device
        CHECK(cudaMemcpy(
            d_X, X, n_input_elements * sizeof(float), 
            cudaMemcpyHostToDevice
        ));
        CHECK(cudaMemcpy(d_W, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Define block and grid size
        dim3 blockSize(32, 32);
        dim3 gridSize((hidden_size + blockSize.x - 1) / blockSize.x,
                        (n_samples + blockSize.y - 1) / blockSize.y);

        // Multiply
        _matmul_GPU<<<gridSize, blockSize>>>(d_X, d_W, d_out, n_samples, n_features, hidden_size);

        // Activation function
        dim3 blockSize_1D(256);
        dim3 gridSize_1D((n_samples * hidden_size + blockSize_1D.x - 1) / 256);
        if (i == Ws.size() - 1)
            _softmax_GPU<<<gridSize_1D, blockSize_1D>>>(d_out, d_out, n_samples, out_size);
        else
            _ReLU_GPU<<<gridSize_1D, blockSize_1D>>>(d_out, n_samples * hidden_size);

        // Copy memory: device-to-host
        CHECK(cudaMemcpy(
            out, d_out, n_output_elements * sizeof(float), 
            cudaMemcpyDeviceToHost
        ));

        timer.Stop();
        time = timer.Elapsed();
        cout << "- layer " << i << " ";
        printf("forward time: %f ms\n", time);

        outs.push_back(out);

        // Free device memory
        CHECK(cudaFree(d_X));
        CHECK(cudaFree(d_W));
        CHECK(cudaFree(d_out));
    }

    return outs;
}


vector<float*> _fw_GPU_optim(vector<float> X, vector<vector<float>> Ws, int n_samples, int n_features, 
                        int hidden_size, int out_size) {

    vector<float*> outs;
    outs.push_back(X.data());
    
    GpuTimer timer;
    float time;


    for (int i = 0; i < Ws.size(); i++) {
        if (i != 0) n_features = hidden_size;
        if (i == Ws.size() - 1) hidden_size = out_size;

        int n_input_elements = (n_samples * n_features);
        int n_output_elements = (n_samples * hidden_size);
 
        timer.Start();

        vector<float> W = Ws[i];
        float *X = outs[i];
        float *out;
        CHECK(cudaMallocHost(&out, n_output_elements * sizeof(float)));

        // Allocate memory on device
        float *d_X, *d_W, *d_out;
        CHECK(cudaMalloc(&d_X, n_input_elements * sizeof(float)));
        CHECK(cudaMalloc(&d_W, W.size() * sizeof(float)));
        CHECK(cudaMalloc(&d_out, n_output_elements * sizeof(float)));

        // Copy memory: host-to-device
        CHECK(cudaMemcpy(
            d_X, X, n_input_elements * sizeof(float), 
            cudaMemcpyHostToDevice
        ));
        CHECK(cudaMemcpy(d_W, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Define block and grid size
        dim3 blockSize(32, 32);
        dim3 gridSize((hidden_size + blockSize.x - 1) / blockSize.x,
                        (n_samples + blockSize.y - 1) / blockSize.y);

        // Multiply
        _tiled_matmul_GPU<<<gridSize, blockSize>>>(d_X, d_W, d_out, n_samples, n_features, hidden_size);

        // Activation function
        dim3 blockSize_1D(256);
        dim3 gridSize_1D((n_samples * hidden_size + blockSize_1D.x - 1) / 256);
        if (i == Ws.size() - 1)
            _softmax_GPU<<<gridSize_1D, blockSize_1D>>>(d_out, d_out, n_samples, out_size);
        else
            _ReLU_GPU<<<gridSize_1D, blockSize_1D>>>(d_out, n_samples * hidden_size);

        // Copy memory: device-to-host
        CHECK(cudaMemcpy(
            out, d_out, n_output_elements * sizeof(float), 
            cudaMemcpyDeviceToHost
        ));

        timer.Stop();
        time = timer.Elapsed();
        cout << "- layer " << i << " ";
        printf("forward time: %f ms\n", time);

        outs.push_back(out);

        // Free device memory
        CHECK(cudaFree(d_X));
        CHECK(cudaFree(d_W));
        CHECK(cudaFree(d_out));
    }

    return outs;
}
