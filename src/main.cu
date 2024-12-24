#include "nn.h"
#include "data.h"


int main(int argc, char** argv) {
    // hidden_size n_epoch lr
    if (argc != 5) {
        cout << argc << endl;
        cout << "ERROR: invoke the program as\n";
        cout << ">>> ./main <#-neurons> <#-epochs> <learning-rate> <mode>\n";
        return 0;
    }

    //-- Set up 
    const int input_size = 784;
    const int hidden_size = atoi(argv[1]);
    const int output_size = 10;
    const int max_epoch = atoi(argv[2]);
    const float learning_rate = atof(argv[3]);

    bool use_gpu = true, optimize = true;
    int mode = atoi(argv[4]);
    if (mode == 1) {
        use_gpu = false;
        optimize = false;
    } else if (mode == 2) {
        use_gpu = true;
        optimize = false;
    }

    cout << "-- # neurons: " << hidden_size << endl;
    cout << "-- # epochs: " << max_epoch << endl;
    cout << "-- learning rate: " << learning_rate << endl;

    //-- Init weights
    vector<vector<float>> Ws;

    vector<float> W1(input_size * hidden_size);
    Ws.push_back(W1);
    vector<float> W2(hidden_size * hidden_size);
    Ws.push_back(W2);
    vector<float> W3(hidden_size * output_size);
    Ws.push_back(W3);

    // Init weights
    init_weights(Ws);

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

    // Flatten input batch
    vector<float> X_fwtest(num_img_train * input_size);
    for (int i = 0; i < num_img_train; ++i) {
        copy(trainImages[i].begin(), trainImages[i].end(), X_fwtest.begin() + i * input_size);
    }

    //-- TEST FORWARD RUNTIME
    // GpuTimer timer;
    // float time;
    // timer.Start();
    // vector<float*> outputs_cpu = forward(
    //     X_fwtest, Ws, num_img_train, input_size, hidden_size, output_size, false, false
    // );
    // timer.Stop();
    // time = timer.Elapsed();
    // printf("FORWARD TIME CPU: %f ms\n\n", time);

    // timer.Start();
    // vector<float*> outputs_gpu = forward(
    //     X_fwtest, Ws, num_img_train, input_size, hidden_size, output_size, true, false
    // );
    // timer.Stop();
    // time = timer.Elapsed();
    // printf("FORWARD TIME GPU: %f ms\n\n", time);

    // timer.Start();
    // vector<float*> outputs_gpu_optim = forward(
    //     X_fwtest, Ws, num_img_train, input_size, hidden_size, output_size, true, true
    // );
    // timer.Stop();
    // time = timer.Elapsed();
    // printf("FORWARD TIME GPU (OPTIMIZED): %f ms\n\n", time);

    // float err = 0;
    // for (int i = 0; i < num_img_train * output_size; i++)
    //     err += abs(outputs_cpu.at(3)[i] - outputs_gpu.at(3)[i]);
    // cout << "-- Mean error CPU - GPU: " << err / (num_img_train * output_size) << endl;

    // err = 0;
    // for (int i = 0; i < num_img_train * output_size; i++)
    //     err += abs(outputs_cpu.at(3)[i] - outputs_gpu_optim.at(3)[i]);
    // cout << "-- Mean error CPU - GPU (optimized): " << err / (num_img_train * output_size) << endl;

    //-- CPU train
    GpuTimer timer;
    float time;
    cout << "\nTrain start...\n";
    cout << "-- number of epochs: " << max_epoch << endl;
    timer.Start();
    train(trainImages, trainLabels, Ws,
           hidden_size, output_size, max_epoch, learning_rate, use_gpu, optimize);
    timer.Stop();
    time = timer.Elapsed();
    printf("TRAIN TIME: %f ms\n\n", time);


    return 0;
}
