#include "nn.h"
#include "data.h"


int main(int argc, char** argv) {
    // hidden_size n_epoch lr
    if (argc != 5) {
        cout << argc << endl;
        cout << "ERROR: invoke the program as\n";
        cout << ">>> ./main <#-neurons> <#-epochs> <learning-rate> <is-optimized>\n";
        return 0;
    }

    //-- Set up 
    const int hidden_size = atoi(argv[1]);
    const int max_epoch = atoi(argv[2]);
    const float learning_rate = atof(argv[3]);

    bool optimize = true;
    if (atoi(argv[4]) == 0) optimize = false;

    cout << "-- # neurons: " << hidden_size << endl;
    cout << "-- # epochs: " << max_epoch << endl;
    cout << "-- learning rate: " << learning_rate << endl;
    cout << "-- optimize GPU (tiled matmul in FW, fp16 in BW): " << optimize << endl;

    //-- Load data
    vector<vector<float>> trainImages;
    vector<int> trainLabels;
    vector<vector<float>> testImages;
    vector<int> testLabels;
    int num_img_train, num_img_test, imageSize, num_label_train, num_label_test, n_rows, n_cols;
    try {
        // Load training images
        readImages("mnist/train-images-idx3-ubyte", trainImages, num_img_train, imageSize, n_rows, n_cols);
        cout << "Train Images: " << num_img_train << " with size " << imageSize << endl;

        // Load training labels
        readLabels("mnist/train-labels-idx1-ubyte", trainLabels, num_label_train);
        cout << "Train Labels: " << num_label_train << " labels loaded" << endl;

        // Load test images
        readImages("mnist/t10k-images-idx3-ubyte", testImages, num_img_test, imageSize, n_rows, n_cols);
        cout << "Test Images: " << num_img_test << " with size " << imageSize << endl;

        // Load test labels
        readLabels("mnist/t10k-labels-idx1-ubyte", testLabels, num_label_test);
        cout << "Test Labels: " << num_label_test << " labels loaded" << endl;

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
    cout << endl;

    const float output_size = 10;

    //-- Init weights
    // TODO: user defined number of layers
    vector<vector<float>> Ws;
    vector<float> W1(imageSize * hidden_size); 
    Ws.push_back(W1);
    vector<float> W2(hidden_size * hidden_size);
    Ws.push_back(W2);
    vector<float> W3(hidden_size * output_size);
    Ws.push_back(W3);

    // Init weights
    init_weights(Ws);

    vector<vector<float>> Ws_gpu = Ws;

    // Flatten input data
    float* X_train = new float[num_img_train * imageSize];
    for (int i = 0; i < num_img_train; i++) {
        for (int j = 0; j < imageSize; j++)
            X_train[i * imageSize + j] = trainImages[i][j];
    }

    GpuTimer timer;

    //-- TEST FORWARD RUNTIME
    float time;
    timer.Start();
    vector<float*> outputs_cpu = forward(
        X_train, Ws, num_img_train, imageSize, hidden_size, output_size, false, false
    );
    timer.Stop();
    time = timer.Elapsed();
    printf("> FORWARD TIME CPU: %f ms\n\n", time);

    timer.Start();
    vector<float*> outputs_gpu = forward(
        X_train, Ws_gpu, num_img_train, imageSize, hidden_size, output_size, true, optimize
    );
    timer.Stop();
    time = timer.Elapsed();
    printf("> FORWARD TIME GPU: %f ms\n\n", time);

    float err = 0;
    for (int i = 0; i < num_img_train * output_size; i++)
        err += abs(outputs_cpu.at(3)[i] - outputs_gpu.at(3)[i]);
    cout << "-- Mean error CPU - GPU: " << err / (num_img_train * output_size) << endl;

    // Train
    cout << "\nTraining on CPU...\n";
    timer.Start();
    train(X_train, trainLabels, Ws,
           num_img_train, imageSize, hidden_size, output_size, 
           max_epoch, learning_rate, false, false);
    timer.Stop();
    time = timer.Elapsed();
    printf("> TRAIN TIME: %f ms\n\n", time);
    
    cout << "Training on GPU...\n";
    timer.Start();
    train(X_train, trainLabels, Ws_gpu,
           num_img_train, imageSize, hidden_size, output_size, 
           max_epoch, learning_rate, true, optimize);
    timer.Stop();
    time = timer.Elapsed();
    printf("> TRAIN TIME: %f ms\n\n", time);

    // Flatten test data
    float* X_test = new float[num_img_test * imageSize];
    for (int i = 0; i < num_img_test; i++) {
        for (int j = 0; j < imageSize; j++)
            X_test[i * imageSize + j] = testImages[i][j];
    }

    cout << "Forward on test set, CPU...\n";
    vector<float*> test_outputs_cpu = forward(
        X_test, Ws, num_img_test, imageSize, hidden_size, output_size, false, false
    );
    cout << "Forward on test set, GPU...\n";
    vector<float*> test_outputs_gpu = forward(
        X_test, Ws_gpu, num_img_test, imageSize, hidden_size, output_size, true, optimize
    );
    err = 0;
    for (int i = 0; i < num_img_test * output_size; i++)
        err += abs(test_outputs_cpu.at(3)[i] - test_outputs_gpu.at(3)[i]);
    cout << "-- Mean error CPU - GPU: " << err / (num_img_test * output_size) << endl;

    return 0;
}
