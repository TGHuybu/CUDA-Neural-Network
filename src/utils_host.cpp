#include "utils_host.h"


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


void init_weights(vector<vector<float>> &Ws) {
    // AMAZING RANDOM GENERATOR (WOW!)
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
    
    // Init weights with random numbers
    for (int i = 0; i < Ws.size(); i++)
        for (auto &w : Ws[i]) w = randValue();
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



