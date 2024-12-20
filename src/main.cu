# include "nn.h"


int main() {
    

    const int batch_size = 32;
    const int input_size = 784;
    const int hidden1_size = 128;
    const int hidden2_size = 128;
    const int output_size = 10;

    // Khởi tạo các trọng số và bias
    vector<float> W1(input_size * hidden1_size);
    vector<float> b1(hidden1_size);
    vector<float> W2(hidden1_size * hidden2_size);
    vector<float> b2(hidden2_size);
    vector<float> W3(hidden2_size * output_size);
    vector<float> b3(output_size);

    init_param(W1, b1, W2, b2, W3, b3);

    // Đọc dữ liệu MNIST
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
        cerr << "Error reading MNIST data: " << e.what() << endl;
        return 1;
    }

    // Sử dụng batch_size ảnh đầu tiên làm input
    vector<float> X(batch_size * input_size);
    for (int i = 0; i < batch_size; ++i) {
        copy(trainImages[i].begin(), trainImages[i].end(), X.begin() + i * input_size);
    }

    // Kết quả đầu ra
    vector<float> output(batch_size * output_size);

    // Gọi forward pass trên GPU
    forwardCUDA(X.data(), W1.data(), b1.data(),
                W2.data(), b2.data(),
                W3.data(), b3.data(),
                output.data(), batch_size);

    // In kết quả đầu ra (chỉ in vài giá trị để kiểm tra)
    cout << "Output from the forward pass (first 10 values):\n";
    for (int i = 0; i < min(10, batch_size * output_size); ++i) {
        cout << output[i] << " ";
    }
    cout << endl;

    return 0;
}