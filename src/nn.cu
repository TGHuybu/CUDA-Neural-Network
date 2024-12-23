#include "nn.h"


vector<float> one_hot(vector<int> y, int n_samples, int n_classes) {
    vector<float> onehots(n_samples * n_classes, 0);
    for (int i = 0; i < y.size(); i++) {
        int idx = n_classes * i + y.at(i);
        onehots[idx] = 1.0;
    }

    return onehots;
}


float loss(float* y_pred, float* y_true, int n_samples, int n_classes) {
    float* abs_err = _add_CPU(y_pred, y_true, n_samples * n_classes, -1);
    float cee = _sum_CPU(abs_err, n_samples * n_classes);
    return (-1 * cee) / n_samples;
}


vector<float*> forward(vector<float> X, vector<vector<float>> Ws, int n_samples, int n_features, 
                        int hidden_size, int out_size, bool use_gpu) {

    vector<float*> outs;
    if (use_gpu) { 
        outs = _fw_GPU(X, Ws, n_samples, n_features, hidden_size, out_size);
    } else {
        // TODO: CPU forward
    }

    return outs;
}

void train(vector<vector<float>> X, vector<int> y, vector<vector<float>> &Ws,
           int hidden_size, int out_size, int max_epoch, float learning_rate, bool use_gpu) {

    cout << "train\n";
    // Preprocess data
    int n_samples = X.size();
    int n_features = X[0].size(); // Number of features in the input data

    // One-hot encoding
    vector<float> y_onehot = one_hot(y, n_samples, out_size);

    // Flatten input data into 1D vector for compatibility
    vector<float> X_train(n_samples * n_features);
    for (int i = 0; i < n_samples; ++i) {
        copy(X[i].begin(), X[i].end(), X_train.begin() + i * n_features);
    }

    // Cross-entropy errors
    vector<float> cees;

    cout << "start\n";

    for (int epoch = 0; epoch < max_epoch; epoch++) {
        // 1. Forward pass
        vector<float*> outputs = forward(X_train, Ws, n_samples, n_features, hidden_size, out_size, use_gpu);
        cout << epoch << endl;

        cout << "d\n";
        // 2. Delta for the output layer
        float* final_output = outputs.back();
        float* final_input = outputs[outputs.size() - 2];
        float* delta_out = _add_CPU(final_output, y_onehot.data(), n_samples * out_size, -1);

        cout << "g\n";
        // 3. Gradient for the output layer
        float* final_input_T = _transpose(final_input, n_samples, hidden_size);
        float* dOut = _matmul_CPU(final_input_T, delta_out, hidden_size, n_samples, out_size);

        cout << "ww\n";
        // 4. Update weights for the output layer
        float* W_out_updated = _add_CPU(Ws.back().data(), dOut, hidden_size * out_size, -learning_rate);
        Ws.back().assign(W_out_updated, W_out_updated + hidden_size * out_size);
        delete[] W_out_updated;

        // Variables for backpropagation
        int n_input_features = hidden_size;
        int n_output_features = hidden_size;
        float* delta_hidden = delta_out;

        cout << "bp\n";
        // 5. Backpropagate through hidden layers
        for (int layer = Ws.size() - 2; layer >= 0; --layer) {
            if (layer == 0) n_input_features = n_features;
            cout << layer << endl;

            // Get next layer weights
            vector<float> W_next = Ws[layer + 1];
            float* W_next_T = _transpose(W_next.data(), n_output_features, n_input_features);
            cout << "t\n";

            // Get layer outputs and apply ReLU derivative
            float* layer_output = outputs[layer + 1];
            float* dReLU = _dReLU_CPU(layer_output, n_samples * n_output_features);
            cout << "drelu\n";

            // Compute delta for current layer
            float* delta_hidden_temp = _matmul_CPU(delta_hidden, W_next_T, n_samples, n_output_features, n_input_features);
            float* delta_hidden_updated = _ewmul_CPU(delta_hidden_temp, dReLU, n_samples * n_input_features);

            delta_hidden = delta_hidden_updated;
            cout << "delta_hidden\n";

            // Gradient for current layer
            float* layer_input = outputs[layer];
            cout << "glaye\n";
            float* layer_input_T = _transpose(layer_input, n_samples, n_input_features);
            cout << "glaye\n";
            float* dHidden = _matmul_CPU(layer_input_T, delta_hidden, n_input_features, n_samples, n_input_features);
            cout << "glaye\n";

            // Update weights for the current layer
            float* W_hidden_updated = _add_CPU(Ws[layer].data(), dHidden, n_input_features * n_output_features, -learning_rate);
            Ws[layer].assign(W_hidden_updated, W_hidden_updated + n_input_features * n_input_features);
            cout << "descent\n";

            // // Update feature sizes for next iteration
            // n_output_features = n_input_features;
            // n_input_features = (layer == 0) ? n_features : hidden_size;
            // cout << "size\n";
        }

        // 6. Calculate and log cross-entropy loss
        float epoch_loss = loss(y_onehot.data(), outputs.back(), n_samples, out_size);
        cees.push_back(epoch_loss);
        cout << "Epoch " << epoch + 1 << ": Loss = " << epoch_loss << endl;

        // Free memory for deltas
        delete[] delta_out;

        // Free output allocations
        for (float* ptr : outputs) {
            delete[] ptr;
        }
    }
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
    _matmul_GPU<<<gridSize1, blockSize>>>(d_X, d_W1, d_Z1, batch_size, input_size, hidden1_size);

    // Thêm bias vào Z1 và áp dụng ReLU
    dim3 block1D((batch_size * hidden1_size + 255) / 256);
    cudaMemcpy(d_Z1, d_b1, hidden1_size * sizeof(float), cudaMemcpyDeviceToDevice);
    _ReLU_GPU<<<block1D, 256>>>(d_Z1, batch_size * hidden1_size);

    // Lớp ẩn thứ hai: Z2 = Z1 * W2 + b2
    dim3 gridSize2((hidden2_size + blockSize.x - 1) / blockSize.x,
                   (batch_size + blockSize.y - 1) / blockSize.y);
    _matmul_GPU<<<gridSize2, blockSize>>>(d_Z1, d_W2, d_Z2, batch_size, hidden1_size, hidden2_size);

    // Thêm bias vào Z2 và áp dụng ReLU
    dim3 block2D((batch_size * hidden2_size + 255) / 256);
    cudaMemcpy(d_Z2, d_b2, hidden2_size * sizeof(float), cudaMemcpyDeviceToDevice);
    _ReLU_GPU<<<block2D, 256>>>(d_Z2, batch_size * hidden2_size);

    // Lớp đầu ra: output = Z2 * W3 + b3
    dim3 gridSize3((output_size + blockSize.x - 1) / blockSize.x,
                   (batch_size + blockSize.y - 1) / blockSize.y);
    _matmul_GPU<<<gridSize3, blockSize>>>(d_Z2, d_W3, d_output, batch_size, hidden2_size, output_size);
    dim3 blockSoftmax(256);
    dim3 gridSoftmax((batch_size * output_size + blockSoftmax.x - 1) / blockSoftmax.x);
    _softmax_GPU<<<gridSoftmax, blockSoftmax>>>(d_output, d_output, batch_size, output_size);

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