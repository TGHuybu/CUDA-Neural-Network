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


float* _transpose(float *A, int n_rows, int n_cols) {
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


float* _softmax_CPU(float *input, int batch_size, int output_size) {
    float* output = new float[batch_size * output_size];
    for (int i = 0; i < batch_size; i++) {

        float local_max = -1;  // const?
        for (int j = 0; j < output_size; j++) {
            local_max = max(local_max, input[output_size * i + j]);
        }

        float exp_sum = 0;
        for (int j = 0; j < output_size; j++) {
            float exp_val = exp(input[output_size * i + j] - local_max);
            output[output_size * i + j] = exp_val;
            exp_sum += exp_val;
        }

        for (int j = 0; j < output_size; j++) {
            output[output_size * i + j] /= exp_sum;
        }
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

