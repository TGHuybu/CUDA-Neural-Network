#include "nn.h"
#include "data.h"


int main() {
    
    //-- Set up 
    const int batch_size = 32;
    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;
    const int n_hidden = 2;

    //-- Init weights
    
    // Khởi tạo các trọng số và bias
    vector<float> W1(input_size * hidden_size);
    vector<float> b1(hidden_size);
    vector<float> W2(hidden_size * hidden_size);
    vector<float> b2(hidden_size);
    vector<float> W3(hidden_size * output_size);
    vector<float> b3(output_size);
    
    // -- NEW INIT
    // init_weights(Ws);

    //-- OLD INIT
    init_param(W1, b1, W2, b2, W3, b3);
    vector<vector<float>> Ws;
    Ws.push_back(W1);
    Ws.push_back(W2);
    Ws.push_back(W3);

    //-- Load data
    vector<vector<float>> trainImages;
    vector<int> trainLabels;
    vector<vector<float>> testImages;
    vector<int> testLabels;
    int numImages, imageSize, numLabels, n_rows, n_cols;

    try {
        // Load training images
        readImages("mnist/train-images-idx3-ubyte", trainImages, numImages, imageSize, n_rows, n_cols);
        cout << "Train Images: " << numImages << " with size " << imageSize << " each." << endl;

        // Load training labels
        readLabels("mnist/train-labels-idx1-ubyte", trainLabels, numLabels);
        cout << "Train Labels: " << numLabels << " labels loaded." << endl;

        // Load test images
        readImages("mnist/t10k-images-idx3-ubyte", testImages, numImages, imageSize, n_rows, n_cols);
        cout << "Test Images: " << numImages << " with size " << imageSize << " each." << endl;

        // Load test labels
        readLabels("mnist/t10k-labels-idx1-ubyte", testLabels, numLabels);
        cout << "Test Labels: " << numLabels << " labels loaded." << endl;

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }

    // Flatten input batch
    vector<float> X(batch_size * input_size);
    for (int i = 0; i < batch_size; ++i) {
        copy(trainImages[i].begin(), trainImages[i].end(), X.begin() + i * input_size);
    }

    //-- TEST NEW FORWARD
    vector<float*> outputs = forward(
        X, Ws, batch_size, input_size, hidden_size, output_size
    );
    
    cout << "Output from the NEW forward pass (first 10 values):\n";
    cout << outputs.size() << endl;
    for (int i = 0; i < min(10, batch_size * output_size); ++i)
        cout << outputs.at(3)[i] << " ";
    cout << endl;

    //-- TEST OLD FORWARD
    vector<float> output(batch_size * output_size);

    // Gọi forward pass trên GPU
    forwardCUDA(X.data(), W1.data(), b1.data(),
                W2.data(), b2.data(),
                W3.data(), b3.data(),
                output.data(), batch_size);

    // In kết quả đầu ra (chỉ in vài giá trị để kiểm tra)
    cout << "Output from the forward pass (first 10 values):\n";
    for (int i = 0; i < min(10, batch_size * output_size); ++i)
        cout << output[i] << " ";
    cout << endl;
    cout << "b3:\n";

    return 0;
}