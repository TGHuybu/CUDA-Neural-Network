#include "data.h"


int _reverse_int(int i) {
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
    magic_number = _reverse_int(magic_number);
    if (magic_number != 2051) {
        throw runtime_error("Invalid magic number in image file.");
    }

    file.read((char*)&numImages, sizeof(numImages));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));
    numImages = _reverse_int(numImages);
    n_rows = _reverse_int(n_rows);
    n_cols = _reverse_int(n_cols);

    imageSize = n_rows * n_cols; 

    images.resize(numImages, vector<float>(imageSize));
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, sizeof(pixel));

            // Normalize
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }

    file.close();
}


void readLabels(const string& filename, vector<int>& labels, int& numLabels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = _reverse_int(magic_number);
    if (magic_number != 2049) {
        throw runtime_error("Invalid magic number in label file.");
    }

    file.read((char*)&numLabels, sizeof(numLabels));
    numLabels = _reverse_int(numLabels);

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
