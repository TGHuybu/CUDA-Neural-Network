#include "libs.h"
#include "utils_host.h"
#include "utils_device.h"


vector<float*> forward(vector<float>, vector<vector<float>>, 
                        int, int, int, int, bool=true);


void forwardCUDA(const float* , const float* , const float* , 
                 const float* , const float* ,
                 const float* , const float* ,
                 float* , int );