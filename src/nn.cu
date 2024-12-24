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
    // log(y_pred)
    float* log_y_pred = new float[n_samples * n_classes];
    for (int i = 0; i < n_samples * n_classes; i++) {
        log_y_pred[i] = log(fmax(y_pred[i], 1e-7));
    }

    float* temp = _ewmul_CPU(log_y_pred, y_true, n_samples * n_classes);
    float cee = _sum_CPU(temp, n_samples * n_classes);
    return (-1.0 * cee) / n_samples;
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

        GpuTimer timer;
        float time;

        int layer_in_size = n_features;
        int layer_out_size = hidden_size;
        for (int i = 0; i < Ws.size(); i++) {
            if (i != 0) layer_in_size = hidden_size;
            if (i == Ws.size() - 1) layer_out_size = out_size;

            timer.Start();
    
            vector<float> W = Ws[i];
            float* X_in = outs[i];

            // Multiply
            float* out = _matmul_CPU(X_in, W.data(), n_samples, layer_in_size, layer_out_size);

            // Activation function
            if (i == Ws.size() - 1)
                out = _softmax_CPU(out, n_samples, out_size);
            else
                out = _ReLU_CPU(out, n_samples * hidden_size);

            timer.Stop();
            time = timer.Elapsed();
            cout << "- layer " << i << " ";
            printf("forward time: %f ms\n", time);

            outs.push_back(out);
        }

        outs[0] = X.data();
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
           int hidden_size, int n_classes, int max_epoch, float learning_rate, bool use_gpu, bool optimize) {
    
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
        vector<float*> outs = forward(X_train, Ws, sample_size, n_data_features, hidden_size, n_classes, use_gpu, true);

        // TODO: Branch out to CPU and GPU backward functions
        vector<float*> grads = _backward_CPU(outs, Ws, y_onehot, sample_size, n_data_features, hidden_size, n_classes);

        // Update weights
        update_weights(Ws, grads, learning_rate);

        float cee = loss(outs.back(), y_onehot.data(), sample_size, n_classes);
        cout << ">>> Epoch " << epoch + 1 << " CEE loss: " << cee << endl;
    }
}