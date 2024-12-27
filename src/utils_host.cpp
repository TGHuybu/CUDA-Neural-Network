#include "utils_host.h"


float _randValue(){
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> distrib(0, 1); 

    float value;
    do {
        value = distrib(gen);
    } while (value < -1.0 || value > 1.0);

    return value;
}


void _makeValue(vector<float> &vt, int h, int w){
    for (int i = 0; i < h*w; i++ ){
        vt[i] = _randValue();
    }
}


void init_weights(vector<vector<float>> &Ws) {
    // AMAZING RANDOM GENERATOR (WOW!)
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> distrib(0, 1);

    auto randValue = [&]() -> float {
        float value;
        do {
            value = distrib(gen);
        } while (value < -1.0 || value > 1.0);
        return static_cast<float>(value);
    };
    
    // Init weights with random numbers
    for (int i = 0; i < Ws.size(); i++)
        for (auto &w : Ws[i]) w = _randValue();
}


void init_param(vector<float> &W1, vector<float> &b1,
                vector<float> &W2, vector<float> &b2,
                vector<float> &W3, vector<float> &b3) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> distrib(0, 1);

    auto randValue = [&]() -> float {
        float value;
        do {
            value = distrib(gen);
        } while (value < -1.0 || value > 1.0);
        return static_cast<float>(value);
    };

    for (auto &w : W1) w = _randValue();
    for (auto &w : b1) w = _randValue();
    for (auto &w : W2) w = _randValue();
    for (auto &w : b2) w = _randValue();
    for (auto &w : W3) w = _randValue();
    for (auto &w : b3) w = _randValue();
}


float* _transpose_CPU(float *A, int n_rows, int n_cols) {
    float* A_T = new float[n_rows * n_cols];
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++)
            A_T[n_rows * j + i] = A[n_cols * i + j];

    }

    return A_T;
}


float* _add_CPU(float* a, float* b, int n, float sign) {
    // Perform addition or subtraction (based on sign)
    // >>> c = a + sign * b 
    // >>> <=> (c = a + b) or (c = a - b)
    float* c = new float[n];
    for (int i = 0; i < n; i++)
        c[i] = a[i] + sign * b[i];

    return c;
}


float* _ewmul_CPU(float* a, float* b, int n) {
    // Element-wise multiplicationn
    float* c = new float[n];
    for (int i = 0; i < n; i++)
        c[i] = a[i] * b[i];

    return c;
}


float* _matmul_CPU(float* A, float* B, int m, int n, int k) {
    float* C = new float[m * k];
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            float c = 0;
            for (int i = 0; i < n; i++) 
                c += A[row * n + i] * B[i * k + col];

            C[row * k + col] = c;
        }
    }

    return C;
}


float* _scalar_div(float* A, int n, float b) {
    float* B = new float[n];
    for (int i = 0; i < n; i++)
        B[i] = A[i] / b;

    return B;
}


float _sum_CPU(float* a, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];

    return sum;
}


float* _ReLU_CPU(float* Z, int size) {
    float* output = new float[size];
    for (int i = 0; i < size; i++)
        output[i] = fmaxf(0, Z[i]);

    return output;
}


float* _softmax_CPU(float *input, int n_samples, int n_classes) {
    float* output = new float[n_samples * n_classes];
    for (int i = 0; i < n_samples; i++) {

        float local_max = -1;  // const?
        for (int j = 0; j < n_classes; j++) 
            local_max = max(local_max, input[n_classes * i + j]);

        float exp_sum = 0;
        for (int j = 0; j < n_classes; j++) {
            float exp_val = exp(input[n_classes * i + j] - local_max);
            output[n_classes * i + j] = exp_val;
            exp_sum += exp_val;
        }

        for (int j = 0; j < n_classes; j++) 
            output[n_classes * i + j] /= exp_sum;
    }

    return output;
}


float* _dReLU_CPU(float* y, int n) {
    float* dy = new float[n];
    for (int i = 0; i < n; i++) {
        if (y[i] >= 0) dy[i] = 1;
        else dy[i] = 0;
    }

    return dy;
}


vector<float*> _backward_CPU(vector<float*> outs, vector<vector<float>> Ws, 
                        vector<float> y_onehot, int n_samples, int n_features, 
                        int hidden_size, int n_classes) {

    vector<float*> gradients(Ws.size());

    // Final output layer error
    // delta_out = final_output - y_onehot
    float* final_output = outs.back();
    float* delta_out = _add_CPU(final_output, y_onehot.data(), n_samples * n_classes, -1);
    float* delta_hidden = delta_out; 

    // Final layer gradient
    float* final_input = outs[outs.size() - 2];  // Input to the final layer
    float* final_input_T = _transpose_CPU(final_input, n_samples, hidden_size);
    float* grad_out = _matmul_CPU(final_input_T, delta_out, hidden_size, n_samples, n_classes);
    grad_out = _scalar_div(grad_out, hidden_size * n_classes, n_samples);

    // Store gradient
    gradients.back() = grad_out; 

    free(final_input_T);

    // BEGIN BACKPROPAGATION
    int layer_input_size = hidden_size;
    int layer_output_size = hidden_size;
    for (int layer = Ws.size() - 2; layer > -1; layer--) {
        if (layer == 0) layer_input_size = n_features;

        // Current layer input + outputs
        float* layer_input = outs[layer];
        float* layer_output = outs[layer + 1];

        // Obtain next layer's weights, input + output sizes
        int next_layer = layer + 1;
        int next_layer_input_size = layer_output_size;
        int next_layer_output_size = hidden_size;
        if (next_layer == Ws.size() - 1) next_layer_output_size = n_classes;
        vector<float> W_next = Ws[next_layer];

        // ReLU derivative
        float* dReLU = _dReLU_CPU(layer_output, n_samples * layer_output_size);

        // Transpose next layer's weights
        float* W_next_T = _transpose_CPU(W_next.data(), next_layer_input_size, next_layer_output_size);

        // Current layer's output error
        float* delta_hidden_temp = _matmul_CPU(delta_hidden, W_next_T, n_samples, next_layer_output_size, next_layer_input_size);
        float* delta_hidden_new = _ewmul_CPU(delta_hidden_temp, dReLU, n_samples * layer_output_size);

        free(delta_hidden);
        free(delta_hidden_temp);
        free(dReLU);

        // Update output error
        delta_hidden = delta_hidden_new;

        float* layer_input_T = _transpose_CPU(layer_input, n_samples, layer_input_size);
        float* grad_hidden = _matmul_CPU(layer_input_T, delta_hidden, layer_input_size, n_samples, layer_output_size);
        grad_hidden = _scalar_div(grad_hidden, layer_input_size * layer_output_size, n_samples);
        gradients[layer] = grad_hidden;
        
        free(layer_input_T);
        free(W_next_T);
    }

    free(delta_hidden);

    return gradients;
}