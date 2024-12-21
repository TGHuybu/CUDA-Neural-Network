#include "utils_host.h"


int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void readImages(const string& filename, vector<vector<float>>& images, int& numImages, int& imageSize, int& n_rows, int& n_cols) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    if (magic_number != 2051) {
        throw runtime_error("Invalid magic number in image file.");
    }

    file.read((char*)&numImages, sizeof(numImages));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));
    numImages = ReverseInt(numImages);
    n_rows = ReverseInt(n_rows);
    n_cols = ReverseInt(n_cols);

    imageSize = n_rows * n_cols; // Set image size to be the number of pixels in one image

    images.resize(numImages, vector<float>(imageSize));
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f; // Normalize to [0, 1]
        }
    }

    file.close();
}


void readLabels(const string& filename, vector<int>& labels, int& numLabels) {
    /*
     * Function: readMNISTLabels
     * This function loads MNIST label data from the specified file.
     * 
     * Data Organization After Loading:
     * - `labels`: A vector of integers where each element represents a single label.
     *   - Size: `labels[numLabels]`, where `numLabels` is the total number of labels.
     * - `numLabels`: Total number of labels loaded.
     */
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    if (magic_number != 2049) {
        throw runtime_error("Invalid magic number in label file.");
    }

    file.read((char*)&numLabels, sizeof(numLabels));
    numLabels = ReverseInt(numLabels);

    labels.resize(numLabels);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    file.close();
}


// Function to save the image as a PNG using stb_image_write
/*
void saveImageAsPNG(const vector<float>& image, const string& filename, int n_rows, int n_cols) {
    // Use n_rows and n_cols for image dimensions
    vector<unsigned char> imgData(n_rows * n_cols);

    // Convert float pixel values (0 to 1) to unsigned char (0 to 255)
    for (int i = 0; i < n_rows; ++i) {  // Iterate over the rows
        for (int j = 0; j < n_cols; ++j) {  // Iterate over the columns
            int index = i * n_cols + j;  // Calculate the index in the 1D array
            imgData[index] = static_cast<unsigned char>(image[index] * 255.0f);  // Convert to unsigned char
        }
    }

    // Save the image using stb_image_write as a grayscale PNG
    int result = stbi_write_png(filename.c_str(), n_cols, n_rows, 1, imgData.data(), n_cols);
    if (result == 0) {
        cerr << "Error saving image to file: " << filename << endl;
    } else {
        cout << "Image saved as: " << filename << endl;
    }
}
*/


float randValue(){
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> distrib(0, 1); 

    float value;
    do {
        value = distrib(gen);
    } while (value < -1.0 || value > 1.0);

    return value;
}


void makeValue(vector<float> &vt, int h, int w){
    for (int i = 0; i < h*w; i++ ){
        vt[i] = randValue();
    }
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

    for (auto &w : W1) w = randValue();
    for (auto &w : b1) w = randValue();
    for (auto &w : W2) w = randValue();
    for (auto &w : b2) w = randValue();
    for (auto &w : W3) w = randValue();
    for (auto &w : b3) w = randValue();
}

// void init_param(){
//     vector<float> W1(784*128);
//     vector<float> b1(1,128);
//     vector<float> W2(128,128);
//     vector<float> b2(1,128);
//     vector<float> W3(128,10);
//     vector<float> b3(10,10);
    
//     makeValue(W1,784,128);
//     makeValue(b1,1,128);
//     makeValue(W2,128,128);
//     makeValue(b2,1,128);
//     makeValue(W3,128,10);
//     makeValue(b3,1,10);

//     return W1,b1,W2,b2,W3,b3;
// }



