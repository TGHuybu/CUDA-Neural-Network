#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

// Helper function to reverse byte order (endian conversion)
int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// Data loader function for images (like read_mnist_data in mnist.cc)
void readMNISTImages(const string& filename, vector<vector<float>>& images, int& numImages, int& imageSize) {
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
    file.read((char*)&imageSize, sizeof(imageSize));
    numImages = ReverseInt(numImages);
    imageSize = ReverseInt(imageSize);

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

// Data loader function for labels (like read_mnist_label in mnist.cc)
void readMNISTLabels(const string& filename, vector<int>& labels, int& numLabels) {
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
