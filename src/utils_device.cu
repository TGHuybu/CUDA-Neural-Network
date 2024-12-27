#include "utils_device.h"


__global__ void _transpose_GPU(float* A, float* A_T, int n_rows, int n_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_rows && j < n_cols) {
        float temp = A[n_cols * i + j];
        temp = fminf(fmaxf(temp, -3.4e38f), 3.4e38f);
        A_T[n_rows * j + i] = temp;
    }
}


__global__ void relu_derivative(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx];
        temp = fminf(fmaxf(temp, -3.4e38f), 3.4e38f);
        output[idx] = (temp >= 0) ? 1.0f : 0.0f;
    }
}


__global__ void scalar_div(float* data, int size, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = data[idx];
        temp = fminf(fmaxf(temp, -3.4e38f), 3.4e38f); 
        data[idx] = temp / scalar;
    }
}


__global__ void _add_GPU(float* A, float* B, float* C, int n, float sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_val = A[idx];
        float b_val = B[idx];
        a_val = fminf(fmaxf(a_val, -3.4e38f), 3.4e38f); 
        b_val = fminf(fmaxf(b_val, -3.4e38f), 3.4e38f); 
        C[idx] = a_val + sign * b_val;
    }
}


__global__ void _ewmul_GPU(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_val = A[idx];
        float b_val = B[idx];
        a_val = fminf(fmaxf(a_val, -3.4e38f), 3.4e38f);
        b_val = fminf(fmaxf(b_val, -3.4e38f), 3.4e38f);
        C[idx] = a_val * b_val;
    }
}


__global__ void _matmul_GPU(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < m && col < k) {
        float value = 0;
        for (int i = 0; i < n; i++)  
            value += A[row * n + i] * B[i * k + col];

        value = fminf(fmaxf(value, -3.4e38f), 3.4e38f);
        C[row * k + col] = value;
    }
}


__global__ void _tiled_matmul_GPU(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float c = 0;

    for (int tile = 0; tile < (n + TILE_WIDTH) / TILE_WIDTH; tile++) {
        int tile_row = tile * TILE_WIDTH + threadIdx.y;
        int tile_col = tile * TILE_WIDTH + threadIdx.x;

        // Default tiles' values
        s_A[threadIdx.y][threadIdx.x] = 0.0f;
        s_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Load values from inputs if in range
        if (row < m && tile_col < n) s_A[threadIdx.y][threadIdx.x] = A[row * n + tile_col];
        if (col < k && tile_row < n) s_B[threadIdx.y][threadIdx.x] = B[tile_row * k + col];

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) 
            c += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < k) {
        c = fminf(fmaxf(c, -3.4e38f), 3.4e38f);
        C[row * k + col] = c;
    }
}


__global__ void _sum_GPU(float* in, float* out, int n) {
    int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
    int i = numElemsBeforeBlk + threadIdx.x;

    for (int stride = blockDim.x; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (i < n && i + stride < n) {
                in[i] += in[i + stride];
            }
        }

        __syncthreads(); // Synchronize within each block
    }

    if (threadIdx.x == 0) {
        float temp = in[numElemsBeforeBlk];
        temp = fminf(fmaxf(temp, -3.4e38f), 3.4e38f); 
        atomicAdd(out, temp);
    }
}


__global__ void _ReLU_GPU(float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Z[idx] = fmaxf(0.0f, Z[idx]);
        Z[idx] = fminf(fmaxf(Z[idx], -3.4e38f), 3.4e38f);
    }
}


__global__ void _softmax_GPU(float* input, float* output, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / output_size;
    int output_idx = idx % output_size;

    if (batch_idx >= batch_size) return;

    float local_max = -3.4e38f;
    for (int i = 0; i < output_size; ++i) 
        local_max = fmaxf(local_max, input[batch_idx * output_size + i]);

    // Calculate exp(input - max) and sum_exp for normalization
    float exp_sum = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        exp_sum += exp(input[batch_idx * output_size + i] - local_max);
    }

    // Normalize and output the result
    float result = exp(input[batch_idx * output_size + output_idx] - local_max) / exp_sum;
    result = fminf(fmaxf(result, 0.0f), 1.0f);
    output[idx] = result;
}


__global__ void _transpose_GPU_FP16(__half* A, __half* A_T, int n_rows, int n_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_rows && j < n_cols) {
        float temp = __half2float(A[n_cols * i + j]);
        temp = fminf(fmaxf(temp, -65504.0f), 65504.0f); 
        A_T[n_rows * j + i] = __float2half(temp);
    }
}


__global__ void relu_derivative_FP16(const __half* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = __half2float(input[idx]);
        temp = fminf(fmaxf(temp, -65504.0f), 65504.0f);
        output[idx] = (temp >= 0) ? __float2half(1.0f) : __float2half(0.0f);
    }
}


__global__ void scalar_div_FP16(__half* data, int size, __half scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = __half2float(data[idx]);
        temp = fminf(fmaxf(temp, -65504.0f), 65504.0f);
        data[idx] = __float2half(temp / __half2float(scalar));
    }
}


__global__ void _add_GPU_FP16(__half* A, __half* B, __half* C, int n, __half sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_val = __half2float(A[idx]);
        float b_val = __half2float(B[idx]);

        a_val = fminf(fmaxf(a_val, -65504.0f), 65504.0f);
        b_val = fminf(fmaxf(b_val, -65504.0f), 65504.0f);

        C[idx] = __float2half(a_val + __half2float(sign) * b_val); 
    }
}


__global__ void _ewmul_GPU_FP16(__half* A, __half* B, __half* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_val = __half2float(A[idx]);
        float b_val = __half2float(B[idx]);

        a_val = fminf(fmaxf(a_val, -65504.0f), 65504.0f);
        b_val = fminf(fmaxf(b_val, -65504.0f), 65504.0f);

        C[idx] = __float2half(a_val * b_val);
    }
}


__global__ void _matmul_GPU_FP16(__half* A, __half* B, __half* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float value = 0;
        for (int i = 0; i < n; i++)  
            value += __half2float(A[row * n + i]) * __half2float(B[i * k + col]);

        value = fminf(fmaxf(value, -65504.0f), 65504.0f);

        C[row * k + col] = __float2half(value);
    }
}


vector<float*> _forward_GPU(float* X, vector<vector<float>> Ws, int n_samples, int n_features, 
                            int n_neurons, int out_size, bool optimize) {

    vector<float*> outs;
    outs.push_back(X);
    
    GpuTimer timer;
    float time;

    dim3 blockSize(32, 32);
    dim3 blockSize_1D(256);

    int layer_in_size = n_features;
    int layer_out_size = n_neurons;
    for (int i = 0; i < Ws.size(); i++) {
        if (i != 0) layer_in_size = n_neurons;
        if (i == Ws.size() - 1) layer_out_size = out_size;

        int n_input_elements = (n_samples * layer_in_size);
        int n_output_elements = (n_samples * layer_out_size);
 
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
        CHECK(cudaMemcpy(d_X, X, n_input_elements * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_W, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Define block and grid size
        dim3 gridSize((layer_out_size + blockSize.x - 1) / blockSize.x,
                        (n_samples + blockSize.y - 1) / blockSize.y);

        // Multiply
        if (optimize)
            _tiled_matmul_GPU<<<gridSize, blockSize>>>(d_X, d_W, d_out, n_samples, layer_in_size, layer_out_size);
        else
            _matmul_GPU<<<gridSize, blockSize>>>(d_X, d_W, d_out, n_samples, layer_in_size, layer_out_size);

        // Activation function
        dim3 gridSize_1D((n_samples * n_neurons + blockSize_1D.x - 1) / blockSize_1D.x);
        if (i == Ws.size() - 1)
            _softmax_GPU<<<gridSize_1D, blockSize_1D>>>(d_out, d_out, n_samples, out_size);
        else
            _ReLU_GPU<<<gridSize_1D, blockSize_1D>>>(d_out, n_samples * n_neurons);

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


vector<float*> _backward_GPU(vector<float*> outs, vector<vector<float>> Ws,
                                vector<float> y_onehot, int n_samples, int n_features,
                                int hidden_size, int n_classes) {
    vector<float*> gradients(Ws.size());
    
    GpuTimer timer;
    float time;

    timer.Start();

    // Fixed blocksizes
    dim3 blockSize(32, 32);
    dim3 blockSize_1D(256);

    // Final output layer error
    // delta_out = final_output - y_onehot
    float* final_output = outs.back();
    float *d_final_output, *d_y_onehot, *d_delta_out;
    CHECK(cudaMalloc(&d_final_output, n_samples * n_classes * sizeof(float)));
    CHECK(cudaMalloc(&d_y_onehot, n_samples * n_classes * sizeof(float)));
    CHECK(cudaMalloc(&d_delta_out, n_samples * n_classes * sizeof(float)));
    
    CHECK(cudaMemcpy(d_final_output, final_output, n_samples * n_classes * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y_onehot, y_onehot.data(), n_samples * n_classes * sizeof(float), cudaMemcpyHostToDevice));
    dim3 gridSize_1D((n_samples * n_classes - 1) / blockSize_1D.x + 1);
    // result: d_delta_out
    _add_GPU<<<gridSize_1D, blockSize_1D>>>(d_final_output, d_y_onehot, d_delta_out, n_samples * n_classes, -1);
    float* d_delta_hidden = d_delta_out;

    // Final layer gradient
    float* final_input = outs[outs.size() - 2]; 
    float *d_final_input, *d_final_input_T, *d_grad_out;
    CHECK(cudaMalloc(&d_final_input, n_samples * hidden_size * sizeof(float)));
    CHECK(cudaMalloc(&d_final_input_T, hidden_size * n_samples * sizeof(float)));
    CHECK(cudaMalloc(&d_grad_out, hidden_size * n_classes * sizeof(float)));

    CHECK(cudaMemcpy(d_final_input, final_input, n_samples * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    dim3 gridSize((hidden_size - 1) / blockSize.x + 1, 
                    (n_samples - 1) / blockSize.y + 1);
    // result: d_final_input_T
    _transpose_GPU<<<gridSize, blockSize>>>(d_final_input, d_final_input_T, n_samples, hidden_size);

    gridSize = dim3((n_classes - 1) / blockSize.x + 1, 
                    (hidden_size - 1) / blockSize.y + 1);
    // result: d_grad_out
    _matmul_GPU<<<gridSize, blockSize>>>(d_final_input_T, d_delta_out, d_grad_out, hidden_size, n_samples, n_classes);

    gridSize_1D = dim3((hidden_size * n_classes - 1) / blockSize_1D.x + 1);
    // result: d_grad_out
    scalar_div<<<gridSize_1D, blockSize_1D>>>(d_grad_out, hidden_size * n_classes, n_samples);

    gradients.back() = new float[hidden_size * n_classes];
    CHECK(cudaMemcpy(gradients.back(), d_grad_out, hidden_size * n_classes * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_final_input));
    CHECK(cudaFree(d_final_input_T));
    CHECK(cudaFree(d_grad_out));

    timer.Stop();
    time = timer.Elapsed();
    cout << "- layer " << gradients.size() - 1 << " ";
    printf("backward time: %f ms\n", time);

    // BEGIN BACKPROPAGATION
    int layer_input_size = hidden_size;
    int layer_output_size = hidden_size;
    for (int layer = Ws.size() - 2; layer > -1; layer--) {
        timer.Start();
        if (layer == 0) layer_input_size = n_features;

        // Current layer input + outputs
        float* layer_input = outs[layer];
        float* layer_output = outs[layer + 1];
        float* d_layer_input, *d_layer_output;
        CHECK(cudaMalloc(&d_layer_input, n_samples * layer_input_size * sizeof(float)));
        CHECK(cudaMalloc(&d_layer_output, n_samples * layer_output_size * sizeof(float)));
        CHECK(cudaMemcpy(d_layer_input, layer_input, n_samples * layer_input_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_layer_output, layer_output, n_samples * layer_output_size * sizeof(float), cudaMemcpyHostToDevice));

        // Obtain next layer's weights, input + output sizes
        int next_layer = layer + 1;
        int next_layer_input_size = layer_output_size;
        int next_layer_output_size = hidden_size;
        if (next_layer == Ws.size() - 1) next_layer_output_size = n_classes;

        vector<float> W_next = Ws[next_layer];
        float *d_W_next;
        CHECK(cudaMalloc(&d_W_next, next_layer_input_size * next_layer_output_size * sizeof(float)));
        CHECK(cudaMemcpy(d_W_next, W_next.data(), next_layer_input_size * next_layer_output_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // ReLU derivative
        float* dReLU;
        CHECK(cudaMalloc(&dReLU, n_samples * layer_output_size * sizeof(float)));
        gridSize_1D = dim3((n_samples * layer_output_size - 1) / blockSize_1D.x + 1);
        relu_derivative<<<gridSize_1D, blockSize_1D>>>(d_layer_output, dReLU, n_samples * layer_output_size);

        // Transpose next layer's weights
        float *d_W_next_T;
        CHECK(cudaMalloc(&d_W_next_T, next_layer_output_size * next_layer_input_size * sizeof(float)));
        gridSize = dim3((next_layer_output_size - 1) / blockSize.x + 1, 
                        (next_layer_input_size - 1) / blockSize.y + 1);
        _transpose_GPU<<<gridSize, blockSize>>>(d_W_next, d_W_next_T, next_layer_input_size, next_layer_output_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete

        // Current layer's output error
        float* d_delta_hidden_temp;
        CHECK(cudaMalloc(&d_delta_hidden_temp, n_samples * next_layer_input_size * sizeof(float)));
        gridSize = dim3((next_layer_input_size - 1) / blockSize.x + 1, 
                        (n_samples - 1) / blockSize.y + 1);
        _matmul_GPU<<<gridSize, blockSize>>>(d_delta_hidden, d_W_next_T, d_delta_hidden_temp, n_samples, next_layer_output_size, next_layer_input_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete

        gridSize_1D = dim3((n_samples * layer_output_size - 1) / blockSize_1D.x + 1);

        // Free old d_delta_hidden and allocate new size
        CHECK(cudaFree(d_delta_hidden));
        CHECK(cudaMalloc(&d_delta_hidden, n_samples * layer_output_size * sizeof(float)));

        _ewmul_GPU<<<gridSize_1D, blockSize_1D>>>(d_delta_hidden_temp, dReLU, d_delta_hidden, n_samples * layer_output_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete

        float* d_layer_input_T;
        CHECK(cudaMalloc(&d_layer_input_T, layer_input_size * n_samples * sizeof(float)));
        gridSize = dim3((layer_input_size - 1) / blockSize.x + 1, 
                        (n_samples - 1) / blockSize.y + 1);
        _transpose_GPU<<<gridSize, blockSize>>>(d_layer_input, d_layer_input_T, n_samples, layer_input_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete
        
        // Grad hidden
        float* d_grad_hidden;
        CHECK(cudaMalloc(&d_grad_hidden, layer_input_size * layer_output_size * sizeof(float)));
        
        gridSize = dim3((layer_output_size - 1) / blockSize.x + 1, 
                        (layer_input_size - 1) / blockSize.y + 1);
        _matmul_GPU<<<gridSize, blockSize>>>(d_layer_input_T, d_delta_hidden, d_grad_hidden, layer_input_size, n_samples, layer_output_size);
        gridSize_1D = dim3((layer_input_size * layer_output_size - 1) / blockSize_1D.x + 1);
        scalar_div<<<gridSize_1D, blockSize_1D>>>(d_grad_hidden, layer_input_size * layer_output_size, n_samples);

        float* grad_hidden = new float[layer_input_size * layer_output_size];
        CHECK(cudaMemcpy(grad_hidden, d_grad_hidden, layer_input_size * layer_output_size * sizeof(float), cudaMemcpyDeviceToHost));
        gradients[layer] = grad_hidden;

        CHECK(cudaFree(d_layer_input));
        CHECK(cudaFree(d_layer_output));
        CHECK(cudaFree(dReLU));
        CHECK(cudaFree(d_W_next));
        CHECK(cudaFree(d_W_next_T));
        CHECK(cudaFree(d_delta_hidden_temp));
        CHECK(cudaFree(d_layer_input_T));
        CHECK(cudaFree(d_grad_hidden));
        
        timer.Stop();
        time = timer.Elapsed();
        cout << "- layer " << layer << " ";
        printf("backward time: %f ms\n", time);
    }

    CHECK(cudaFree(d_final_output));
    CHECK(cudaFree(d_y_onehot));
    CHECK(cudaFree(d_delta_hidden));

    return gradients;
}


vector<float*> _backward_GPU_FP16(vector<float*> outs_FP32, vector<vector<float>> Ws_FP32,
                                    vector<float> y_onehot_FP32, int n_samples, int n_features,
                                    int hidden_size, int n_classes) {

    vector<__half*> gradients(Ws_FP32.size());

    GpuTimer timer;
    float time;

    const float loss_scale = 1024.0f;
    //-- CONVERT INPUTS INTO FP16
    // outs_FP32 -> FP16
    vector<__half*> outs(outs_FP32.size());
    for (size_t i = 0; i < outs_FP32.size(); ++i) {
        float* out_FP32 = outs_FP32[i];

        // Size of input array
        int size = n_samples * hidden_size;
        if (i == 0) size = n_samples * n_features;
        else if (i == outs_FP32.size() - 1) size = n_samples * n_classes;

        __half* out_FP16;
        CHECK(cudaMallocHost(&out_FP16, size * sizeof(__half)));
        for (int j = 0; j < size; ++j)
            out_FP16[j] = __float2half(fminf(fmaxf(loss_scale * out_FP32[j], -65504.0f), 65504.0f));

        outs[i] = out_FP16;
    }

    // Ws_FP32 -> FP16
    vector<__half*> Ws(Ws_FP32.size());
    for (size_t i = 0; i < Ws_FP32.size(); ++i) {
        vector<float>& W_FP32 = Ws_FP32[i];
        
        // Size of input array
        int size = hidden_size * hidden_size;
        if (i == 0) size = n_features * hidden_size;
        else if (i == outs_FP32.size() - 1) size = hidden_size * n_classes;

        __half* W_FP16;
        CHECK(cudaMallocHost(&W_FP16, size * sizeof(__half)));
        for (int j = 0; j < size; ++j)
            W_FP16[j] = __float2half(fminf(fmaxf(loss_scale * W_FP32[j], -65504.0f), 65504.0f));

        Ws[i] = W_FP16;
    }
    
    // y_onehot_FP32 -> FP16
    int y_onehot_size = n_samples * n_classes;
    __half* y_onehot;
    CHECK(cudaMallocHost(&y_onehot, y_onehot_size * sizeof(__half)));
    for (int i = 0; i < y_onehot_size; ++i) 
        y_onehot[i] = __float2half(fminf(fmaxf(loss_scale * y_onehot_FP32[i], -65504.0f), 65504.0f));

    timer.Start();

    // Fixed blocksizes
    dim3 blockSize(32, 32);
    dim3 blockSize_1D(256);

    // Final output layer error
    // delta_out = final_output - y_onehot
    __half* final_output = outs.back();
    __half *d_final_output, *d_y_onehot, *d_delta_out;
    CHECK(cudaMalloc(&d_final_output, n_samples * n_classes * sizeof(__half)));
    CHECK(cudaMalloc(&d_y_onehot, n_samples * n_classes * sizeof(__half)));
    CHECK(cudaMalloc(&d_delta_out, n_samples * n_classes * sizeof(__half)));
    
    CHECK(cudaMemcpy(d_final_output, final_output, n_samples * n_classes * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y_onehot, y_onehot, n_samples * n_classes * sizeof(__half), cudaMemcpyHostToDevice));
    dim3 gridSize_1D((n_samples * n_classes - 1) / blockSize_1D.x + 1);
    // result: d_delta_out
    _add_GPU_FP16<<<gridSize_1D, blockSize_1D>>>(d_final_output, d_y_onehot, d_delta_out, n_samples * n_classes, -1);
    __half* d_delta_hidden = d_delta_out;

    // Final layer gradient
    __half* final_input = outs[outs.size() - 2]; 
    __half *d_final_input, *d_final_input_T, *d_grad_out;
    CHECK(cudaMalloc(&d_final_input, n_samples * hidden_size * sizeof(__half)));
    CHECK(cudaMalloc(&d_final_input_T, hidden_size * n_samples * sizeof(__half)));
    CHECK(cudaMalloc(&d_grad_out, hidden_size * n_classes * sizeof(__half)));

    CHECK(cudaMemcpy(d_final_input, final_input, n_samples * hidden_size * sizeof(__half), cudaMemcpyHostToDevice));
    dim3 gridSize((hidden_size - 1) / blockSize.x + 1, 
                    (n_samples - 1) / blockSize.y + 1);
    // result: d_final_input_T
    _transpose_GPU_FP16<<<gridSize, blockSize>>>(d_final_input, d_final_input_T, n_samples, hidden_size);

    gridSize = dim3((n_classes - 1) / blockSize.x + 1, 
                    (hidden_size - 1) / blockSize.y + 1);
    // result: d_grad_out
    _matmul_GPU_FP16<<<gridSize, blockSize>>>(d_final_input_T, d_delta_out, d_grad_out, hidden_size, n_samples, n_classes);

    gridSize_1D = dim3((hidden_size * n_classes - 1) / blockSize_1D.x + 1);
    // result: d_grad_out
    scalar_div_FP16<<<gridSize_1D, blockSize_1D>>>(d_grad_out, hidden_size * n_classes, n_samples);

    gradients.back() = new __half[hidden_size * n_classes];
    CHECK(cudaMemcpy(gradients.back(), d_grad_out, hidden_size * n_classes * sizeof(__half), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_final_input));
    CHECK(cudaFree(d_final_input_T));
    CHECK(cudaFree(d_grad_out));

    timer.Stop();
    time = timer.Elapsed();
    cout << "- layer " << gradients.size() - 1 << " ";
    printf("FP16 backward time: %f ms\n", time);

    // BEGIN BACKPROPAGATION
    int layer_input_size = hidden_size;
    int layer_output_size = hidden_size;
    for (int layer = Ws.size() - 2; layer > -1; layer--) {
        timer.Start();
        if (layer == 0) layer_input_size = n_features;

        // Current layer input + outputs
        __half* layer_input = outs[layer];
        __half* layer_output = outs[layer + 1];
        __half* d_layer_input, *d_layer_output;
        CHECK(cudaMalloc(&d_layer_input, n_samples * layer_input_size * sizeof(__half)));
        CHECK(cudaMalloc(&d_layer_output, n_samples * layer_output_size * sizeof(__half)));
        CHECK(cudaMemcpy(d_layer_input, layer_input, n_samples * layer_input_size * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_layer_output, layer_output, n_samples * layer_output_size * sizeof(__half), cudaMemcpyHostToDevice));

        // Obtain next layer's weights, input + output sizes
        int next_layer = layer + 1;
        int next_layer_input_size = layer_output_size;
        int next_layer_output_size = hidden_size;
        if (next_layer == Ws.size() - 1) next_layer_output_size = n_classes;

        __half* W_next = Ws[next_layer];
        __half *d_W_next;
        CHECK(cudaMalloc(&d_W_next, next_layer_input_size * next_layer_output_size * sizeof(__half)));
        CHECK(cudaMemcpy(d_W_next, W_next, next_layer_input_size * next_layer_output_size * sizeof(__half), cudaMemcpyHostToDevice));
        
        // ReLU derivative
        __half* dReLU;
        CHECK(cudaMalloc(&dReLU, n_samples * layer_output_size * sizeof(__half)));
        gridSize_1D = dim3((n_samples * layer_output_size - 1) / blockSize_1D.x + 1);
        relu_derivative_FP16<<<gridSize_1D, blockSize_1D>>>(d_layer_output, dReLU, n_samples * layer_output_size);

        // Transpose next layer's weights
        __half *d_W_next_T;
        CHECK(cudaMalloc(&d_W_next_T, next_layer_output_size * next_layer_input_size * sizeof(__half)));
        gridSize = dim3((next_layer_output_size - 1) / blockSize.x + 1, 
                        (next_layer_input_size - 1) / blockSize.y + 1);
        _transpose_GPU_FP16<<<gridSize, blockSize>>>(d_W_next, d_W_next_T, next_layer_input_size, next_layer_output_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete

        // Current layer's output error
        __half* d_delta_hidden_temp;
        CHECK(cudaMalloc(&d_delta_hidden_temp, n_samples * next_layer_input_size * sizeof(__half)));
        gridSize = dim3((next_layer_input_size - 1) / blockSize.x + 1, 
                        (n_samples - 1) / blockSize.y + 1);
        _matmul_GPU_FP16<<<gridSize, blockSize>>>(d_delta_hidden, d_W_next_T, d_delta_hidden_temp, n_samples, next_layer_output_size, next_layer_input_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete

        gridSize_1D = dim3((n_samples * layer_output_size - 1) / blockSize_1D.x + 1);

        // Free old d_delta_hidden and allocate new size
        CHECK(cudaFree(d_delta_hidden));
        CHECK(cudaMalloc(&d_delta_hidden, n_samples * layer_output_size * sizeof(__half)));

        _ewmul_GPU_FP16<<<gridSize_1D, blockSize_1D>>>(d_delta_hidden_temp, dReLU, d_delta_hidden, n_samples * layer_output_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete

        __half* d_layer_input_T;
        CHECK(cudaMalloc(&d_layer_input_T, layer_input_size * n_samples * sizeof(__half)));
        gridSize = dim3((layer_input_size - 1) / blockSize.x + 1, 
                        (n_samples - 1) / blockSize.y + 1);
        _transpose_GPU_FP16<<<gridSize, blockSize>>>(d_layer_input, d_layer_input_T, n_samples, layer_input_size);
        CHECK(cudaGetLastError());  // Checks for kernel errors
        CHECK(cudaDeviceSynchronize());  // Ensures all operations are complete
        
        // Grad hidden
        __half* d_grad_hidden;
        CHECK(cudaMalloc(&d_grad_hidden, layer_input_size * layer_output_size * sizeof(__half)));
        
        gridSize = dim3((layer_output_size - 1) / blockSize.x + 1, 
                        (layer_input_size - 1) / blockSize.y + 1);
        _matmul_GPU_FP16<<<gridSize, blockSize>>>(d_layer_input_T, d_delta_hidden, d_grad_hidden, layer_input_size, n_samples, layer_output_size);
        gridSize_1D = dim3((layer_input_size * layer_output_size - 1) / blockSize_1D.x + 1);
        scalar_div_FP16<<<gridSize_1D, blockSize_1D>>>(d_grad_hidden, layer_input_size * layer_output_size, n_samples);

        __half* grad_hidden;
         CHECK(cudaMallocHost(&grad_hidden, layer_input_size * layer_output_size * sizeof(__half)));
        CHECK(cudaMemcpy(grad_hidden, d_grad_hidden, layer_input_size * layer_output_size * sizeof(__half), cudaMemcpyDeviceToHost));
        gradients[layer] = grad_hidden;

        CHECK(cudaFree(d_layer_input));
        CHECK(cudaFree(d_layer_output));
        CHECK(cudaFree(dReLU));
        CHECK(cudaFree(d_W_next));
        CHECK(cudaFree(d_W_next_T));
        CHECK(cudaFree(d_delta_hidden_temp));
        CHECK(cudaFree(d_layer_input_T));
        CHECK(cudaFree(d_grad_hidden));

        timer.Stop();
        time = timer.Elapsed();
        cout << "- layer " << layer << " ";
        printf("FP16 backward time: %f ms\n", time);
    }

    CHECK(cudaFree(d_final_output));
    CHECK(cudaFree(d_y_onehot));
    CHECK(cudaFree(d_delta_hidden));

    vector<float*> gradients_FP32(gradients.size());
    for (size_t i = 0; i < gradients.size(); ++i) {
        __half* grads_FP16 = gradients[i];
        
        // Size of input array
        int size = hidden_size * hidden_size;
        if (i == 0) size = n_features * hidden_size;
        else if (i == gradients.size() - 1) size = hidden_size * n_classes;

        float* grad_FP32;
        CHECK(cudaMallocHost(&grad_FP32, size * sizeof(float)));
        for (int j = 0; j < size; ++j)
            grad_FP32[j] = __half2float(grads_FP16[j]) / loss_scale;

        gradients_FP32[i] = grad_FP32;
    }

    return gradients_FP32;
}
