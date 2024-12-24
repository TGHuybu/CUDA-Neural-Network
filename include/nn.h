#ifndef UTILS_NN_H
#define UTILS_NN_H

#include "libs.h"
#include "utils_host.h"
#include "utils_device.h"


vector<float> one_hot(vector<int>, int, int);

float loss(float*, float*, int, int);

vector<float*> forward(vector<float>, vector<vector<float>>, 
                        int, int, int, int, bool=true, bool=true);

<<<<<<< HEAD
vector<float*> backward(vector<float*> , vector<vector<float>> ,
                        vector<float> , int , int ,
                        int , int , bool );

=======
>>>>>>> 7a83e604914487293214df2b1ebf5b54151e3d16
void train(vector<vector<float>>, vector<int>, vector<vector<float>> &,
           int, int, int, float, bool=true, bool=true);
                 
#endif