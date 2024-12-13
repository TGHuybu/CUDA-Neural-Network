# include "nn.h"


int main() {
    try {
        vector<vector<float>> trainImages;
        vector<int> trainLabels;
         int numImages, imageSize, numLabels,n_rows, n_cols;

        // Load training images
        readImages("mnist/t10k-images-idx3-ubyte", trainImages, numImages, imageSize, n_rows, n_cols);
        cout << "Train Images: " << numImages << " with size " << imageSize << " each." << endl;

        // Load training labels
        readLabels("fashion/t10k-labels-idx1-ubyte", trainLabels, numLabels);
        cout << "Train Labels: " << numLabels << " labels loaded." << endl;

        // Print a sample
        cout << "First train image pixel value: " << trainImages[0][0] << endl;
        cout << "First train label: " << trainLabels[0] << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
