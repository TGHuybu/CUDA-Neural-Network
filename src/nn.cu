#include "nn.h"


vector<float> one_hot(vector<int> y, int n_samples, int n_classes) {
    vector<float> onehots(n_samples * n_classes, 0);
    for (int i = 0; i < y.size(); i++) {
        int idx = n_classes * i + y.at(i);
        onehots[idx] = 1.0;
    }

    return onehots;
}


float mean_binary_error(float* y_pred, int* y_true, int n_samples, int n_classes) {
    int error_count = 0;

    for (int i = 0; i < n_samples; i++) {
        // Find the predicted class
        int predicted_class = 0;
        float max_prob = y_pred[i * n_classes];
        for (int j = 1; j < n_classes; j++) {
            if (y_pred[i * n_classes + j] > max_prob) {
                max_prob = y_pred[i * n_classes + j];
                predicted_class = j;
            }
        }

        // Compare with the true class
        if (predicted_class != y_true[i])
            error_count++;
    }

    return (static_cast<float>(error_count) / n_samples) * 100.0f;
}



float loss(float* y_pred, float* y_true, int n_samples, int n_classes) {
    // log(y_pred)
    float* log_y_pred = new float[n_samples * n_classes];
    for (int i = 0; i < n_samples * n_classes; i++) {
        log_y_pred[i] = log(fmax(y_pred[i], 1e-7));
    }

    float* temp = _ewmul_CPU(log_y_pred, y_true, n_samples * n_classes);
    float cee = _sum_CPU(temp, n_samples * n_classes);
    return (-1 * cee) / n_samples;
}


vector<float*> forward(vector<float> X, vector<vector<float>> Ws, int n_samples, int n_features, 
                        int hidden_size, int out_size, bool use_gpu, bool optimize) {

    vector<float*> outs;
    if (use_gpu) { 
        if (optimize) {
            outs = _fw_GPU_optim(X, Ws, n_samples, n_features, hidden_size, out_size);

            // Set first output as input data
            outs[0] = X.data();
        } else 
            outs = _fw_GPU(X, Ws, n_samples, n_features, hidden_size, out_size);

            // Set first output as input data
            outs[0] = X.data();
    } else {
        //-- Forward using CPU
        outs.push_back(X.data());

        int layer_in_size = n_features;
        int layer_out_size = hidden_size;
        for (int i = 0; i < Ws.size(); i++) {
            if (i != 0) layer_in_size = hidden_size;
            if (i == Ws.size() - 1) layer_out_size = out_size;
    
            vector<float> W = Ws[i];
            float* X_in = outs[i];

            // Multiply
            float* out = _matmul_CPU(X_in, W.data(), n_samples, layer_in_size, layer_out_size);

            // Activation function
            if (i == Ws.size() - 1)
                out = _softmax_CPU(out, n_samples, out_size);
            else
                out = _ReLU_CPU(out, n_samples * hidden_size);

            outs.push_back(out);
        }
    }

    return outs;
}


void update_weights(vector<vector<float>> &Ws, vector<float*> gradients, 
                    float learning_rate) {

    for (int i = 0; i < Ws.size(); i++) {
        for (int j = 0; j < Ws[i].size(); j++)
            Ws[i][j] -= learning_rate * gradients[i][j];
    }
}


void train(vector<vector<float>> X, vector<int> y, vector<vector<float>> &Ws,
           int hidden_size, int n_classes, int max_epoch, float learning_rate, bool use_gpu) {
    
    int sample_size = X.size();
    int n_data_features = X.at(0).size();

    // One-hot encoding
    vector<float> y_onehot = one_hot(y, sample_size, n_classes);

    // Flatten input data
    vector<float> X_train(sample_size * n_data_features);
    for (int i = 0; i < sample_size; ++i)
        copy(X[i].begin(), X[i].end(), X_train.begin() + i * n_data_features);

    for (int epoch = 0; epoch < max_epoch; epoch++) {
        // Forward
        vector<float*> outs = forward(X_train, Ws, sample_size, n_data_features, hidden_size, n_classes, use_gpu, false);

        // TODO: Branch out to CPU and GPU backward functions
        vector<float*> grads = _backward_CPU(outs, Ws, y_onehot, sample_size, n_data_features, hidden_size, n_classes);
        
        // Update weights
        update_weights(Ws, grads, learning_rate);

        float cee = loss(outs.back(), y_onehot.data(), sample_size, n_classes);
        float mbe = mean_binary_error(outs.back(), y.data(), sample_size, n_classes);
        cout << ">>> Epoch " << epoch << " loss: " << cee << "/" << mbe << endl;
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