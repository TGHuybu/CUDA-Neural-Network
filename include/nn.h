#include "libs.h"
#include "utils_host.h"
#include "utils_device.h"


vector<float> one_hot(vector<int>, int, int);

float loss(float*, float*, int, int);

vector<float*> forward(vector<float>, vector<vector<float>>, 
                        int, int, int, int, bool=true, bool=true);

vector<float*> backward(vector<float*>, vector<vector<float>>, 
                        vector<float>, int, int, 
                        int, int);

void train(vector<vector<float>>, vector<int>, vector<vector<float>> &,
           int, int, int, float, bool=true);

void forwardCUDA(const float* , const float* , const float* , 
                 const float* , const float* ,
                 const float* , const float* ,
                 float* , int );