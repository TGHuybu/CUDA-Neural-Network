#include "nn.h"
#include "data.h"


int main() {
    
    //-- Set up 
    const int batch_size = 32;
    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;

    //-- Init weights
    
    // // Khởi tạo các trọng số và bias
    // vector<float> W1(input_size * hidden_size);
    // vector<float> b1(hidden_size);
    // vector<float> W2(hidden_size * hidden_size);
    // vector<float> b2(hidden_size);
    // vector<float> W3(hidden_size * output_size);
    // vector<float> b3(output_size);
    
    // -- NEW INIT
    vector<vector<float>> Ws;

    vector<float> W1(input_size * hidden_size);
    Ws.push_back(W1);

    vector<float> W2(hidden_size * hidden_size);
    Ws.push_back(W2);

    vector<float> W3(hidden_size * output_size);
    Ws.push_back(W3);

    init_weights(Ws);

    // //-- OLD INIT
    // init_param(W1, b1, W2, b2, W3, b3);
    // vector<vector<float>> Ws;
    // Ws.push_back(W1);
    // Ws.push_back(W2);
    // Ws.push_back(W3);

    //-- Load data
    vector<vector<float>> trainImages;
    vector<int> trainLabels;
    vector<vector<float>> testImages;
    vector<int> testLabels;
    int num_img_train, num_img_test, imageSize, num_label_train, num_label_test, n_rows, n_cols;
    try {
        // Load training images
        readImages("mnist/train-images-idx3-ubyte", trainImages, num_img_train, imageSize, n_rows, n_cols);
        cout << "Train Images: " << num_img_train << " with size " << imageSize << " each." << endl;

        // Load training labels
        readLabels("mnist/train-labels-idx1-ubyte", trainLabels, num_label_train);
        cout << "Train Labels: " << num_label_train << " labels loaded." << endl;

        // Load test images
        readImages("mnist/t10k-images-idx3-ubyte", testImages, num_img_test, imageSize, n_rows, n_cols);
        cout << "Test Images: " << num_img_test << " with size " << imageSize << " each." << endl;

        // Load test labels
        readLabels("mnist/t10k-labels-idx1-ubyte", testLabels, num_label_test);
        cout << "Test Labels: " << num_label_test << " labels loaded." << endl;

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }

    // Flatten input batch
    vector<float> X(num_img_train * input_size);
    for (int i = 0; i < num_img_train; ++i) {
        copy(trainImages[i].begin(), trainImages[i].end(), X.begin() + i * input_size);
    }

    //-- TEST NEW FORWARD
    GpuTimer timer;
    timer.Start();
    vector<float*> outputs_cpu = forward(
        X, Ws, num_img_train, input_size, hidden_size, output_size, false
    );
    timer.Stop();
    float time = timer.Elapsed();
    printf("FORWARD TIME CPU: %f ms\n\n", time);

    timer.Start();
    vector<float*> outputs_gpu = forward(
        X, Ws, num_img_train, input_size, hidden_size, output_size, true
    );
    timer.Stop();
    time = timer.Elapsed();
    printf("FORWARD TIME GPU: %f ms\n\n", time);

    float err = 0;
    for (int i = 0; i < num_img_train * output_size; i++) {
        err += outputs_cpu.at(3)[i] - outputs_gpu.at(3)[i]
    }

    //-- Test train
    // train(trainImages, trainLabels, Ws,
    //        hidden_size, output_size, 10, 0.05, true);

    return 0;
}