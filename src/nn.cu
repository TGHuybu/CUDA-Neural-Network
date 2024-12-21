#include "nn.h"


vector<float*> forward(vector<float> X, vector<vector<float>> Ws, 
                        int n_samples, int n_features, int hidden_size, int out_size, bool use_gpu) {
    
    // Convert to pointer for ease of handling
    float *X_in = X.data();
    
    // Save outputs of each layer
    vector<float*> outs;
    outs.push_back(X_in);

    for (int i = 0; i < Ws.size(); i++) {
        if (i != 0) n_features = hidden_size;
        if (i == Ws.size() - 1) hidden_size = out_size;

        vector<float> W = Ws[i];
        float *out = new float[n_samples * hidden_size];
        cout << n_samples * hidden_size << endl;

        if (use_gpu) {
            // Allocate memory on device
            float *d_X, *d_W, *d_out;
            CHECK(cudaMalloc(&d_X, n_samples * n_features *sizeof(float)));
            CHECK(cudaMalloc(&d_W, W.size() * sizeof(float)));
            CHECK(cudaMalloc(&d_out, n_samples * hidden_size * sizeof(float)));

            // Copy memory: host-to-device
            CHECK(cudaMemcpy(d_X, X_in, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_W, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));

            // Define block and grid size
            dim3 blockSize(16, 16);
            dim3 gridSize((hidden_size + blockSize.x - 1) / blockSize.x,
                            (n_samples + blockSize.y - 1) / blockSize.y);

            // Multiply
            matMul<<<gridSize, blockSize>>>(d_X, d_W, d_out, n_samples, n_features, hidden_size);

            // Activation function
            dim3 blockSize_1D(256);
            dim3 gridSize_1D((n_samples * hidden_size + blockSize_1D.x - 1) / 256);
            if (i == Ws.size() - 1)
                softmax<<<gridSize_1D, blockSize_1D>>>(d_out, d_out, n_samples, out_size);
            else
                ReLU<<<gridSize_1D, blockSize_1D>>>(d_out, n_samples * hidden_size);

            // Copy memory: device-to-host
            CHECK(cudaMemcpy(out, d_out, n_samples * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

            // Free device memory
            CHECK(cudaFree(d_X));
            CHECK(cudaFree(d_W));
            CHECK(cudaFree(d_out));

        } else {
            // TODO: CPU forward
        }

        outs.push_back(out);
        X_in = out;
    }

    return outs;
}


void forwardCUDA(const float* h_X, const float* h_W1, const float* h_b1, 
                 const float* h_W2, const float* h_b2,
                 const float* h_W3, const float* h_b3,
                 float* h_output, int batch_size) {

    const int input_size = 784;
    const int hidden1_size = 128;
    const int hidden2_size = 128;
    const int output_size = 10;

    float *d_X, *d_W1, *d_b1, *d_Z1, *d_A1;
    float *d_W2, *d_b2, *d_Z2, *d_A2;
    float *d_W3, *d_b3, *d_Z3, *d_output;

    size_t size_X = batch_size * input_size * sizeof(float);
    size_t size_hidden1 = batch_size * hidden1_size * sizeof(float);
    size_t size_hidden2 = batch_size * hidden2_size * sizeof(float);
    size_t size_output = batch_size * output_size * sizeof(float);

    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W1, input_size * hidden1_size * sizeof(float));
    cudaMalloc(&d_b1, hidden1_size * sizeof(float));
    cudaMalloc(&d_Z1, size_hidden1);
    cudaMalloc(&d_A1, size_hidden1);

    cudaMalloc(&d_W2, hidden1_size * hidden2_size * sizeof(float));
    cudaMalloc(&d_b2, hidden2_size * sizeof(float));
    cudaMalloc(&d_Z2, size_hidden2);
    cudaMalloc(&d_A2, size_hidden2);

    cudaMalloc(&d_W3, hidden2_size * output_size * sizeof(float));
    cudaMalloc(&d_b3, output_size * sizeof(float));
    cudaMalloc(&d_Z3, size_output);
    cudaMalloc(&d_output, size_output);

    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, input_size * hidden1_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, hidden1_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, hidden1_size * hidden2_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, hidden2_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, h_W3, hidden2_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Block và Grid size
    dim3 blockSize(16, 16);
    dim3 gridSize1((hidden1_size + blockSize.x - 1) / blockSize.x,
                   (batch_size + blockSize.y - 1) / blockSize.y);

    // Lớp ẩn thứ nhất: Z1 = X * W1 + b1
    matMul<<<gridSize1, blockSize>>>(d_X, d_W1, d_Z1, batch_size, input_size, hidden1_size);

    // Thêm bias vào Z1 và áp dụng ReLU
    dim3 block1D((batch_size * hidden1_size + 255) / 256);
    cudaMemcpy(d_Z1, d_b1, hidden1_size * sizeof(float), cudaMemcpyDeviceToDevice);
    ReLU<<<block1D, 256>>>(d_Z1, batch_size * hidden1_size);

    // Lớp ẩn thứ hai: Z2 = Z1 * W2 + b2
    dim3 gridSize2((hidden2_size + blockSize.x - 1) / blockSize.x,
                   (batch_size + blockSize.y - 1) / blockSize.y);
    matMul<<<gridSize2, blockSize>>>(d_Z1, d_W2, d_Z2, batch_size, hidden1_size, hidden2_size);

    // Thêm bias vào Z2 và áp dụng ReLU
    dim3 block2D((batch_size * hidden2_size + 255) / 256);
    cudaMemcpy(d_Z2, d_b2, hidden2_size * sizeof(float), cudaMemcpyDeviceToDevice);
    ReLU<<<block2D, 256>>>(d_Z2, batch_size * hidden2_size);

    // Lớp đầu ra: output = Z2 * W3 + b3
    dim3 gridSize3((output_size + blockSize.x - 1) / blockSize.x,
                   (batch_size + blockSize.y - 1) / blockSize.y);
    matMul<<<gridSize3, blockSize>>>(d_Z2, d_W3, d_output, batch_size, hidden2_size, output_size);
    dim3 blockSoftmax(256);
    dim3 gridSoftmax((batch_size * output_size + blockSoftmax.x - 1) / blockSoftmax.x);
    softmax<<<gridSoftmax, blockSoftmax>>>(d_output, d_output, batch_size, output_size);

    // Copy kết quả từ GPU về CPU
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Giải phóng bộ nhớ
    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_Z1);
    cudaFree(d_A1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_Z2);
    cudaFree(d_A2);
    cudaFree(d_W3);
    cudaFree(d_b3);
    cudaFree(d_Z3);
    cudaFree(d_output);
}

// void foward( float* X, float* W1, float* W2, float* W3, float* b1, float* b2, float* b3,
//             int n_input){

    
//     vector<vector<float>> Z1(n_input, 128);
//     vector<vector<float>> A1(n_input,128);
//     vector<vector<float>> Z2(n_input,128);
//     vector<vector<float>> A2(n_input,128);
//     vector<vector<float>> Z3(n_input,10);
//     vector<vector<float>> A3(n_input,1);
//    	dim3 blockSize(32, 32); // Default


//     matMul(X, W1, Z1, n_input, 784,128, blockSize);
//     A1 = ReLU(Z1, n_input);
//     matMul(A1, W2, Z2, n_input, 128,128, blockSize);
//     A2 = ReLU(Z2, n_input);
//     matMul(A2,W3, Z3, n_input, 128,10, blockSize);
//     A3 = softmax(Z3, n_input);

//     return 0;
// }