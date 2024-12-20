#include "libs.h"
#include "utils_host.h"
#include "utils_device.h"

void forwardCUDA(const float* , const float* , const float* , 
                 const float* , const float* ,
                 const float* , const float* ,
                 float* , int );