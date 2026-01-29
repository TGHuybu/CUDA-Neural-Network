#ifndef UTILS_HOST_H
#define UTILS_HOST_H

#include "libs.h"

void init_mat(vector<vector<float>> &mat);

void init_arr_int(vector<int> &arr);

float* _transpose_CPU(float*, int, int);

float* _add_CPU(float*, float*, int, float);

float* _ewmul_CPU(float*, float*, int);

float* _matmul_CPU(float*, float*, int, int, int);

float* _scalar_div(float*, int, float);

float _sum_CPU(float*, int);

float* _ReLU_CPU(float*, int);

float* _softmax_CPU(float*, int, int);

float* _dReLU_CPU(float*, int);

vector<float*> _backward_CPU(vector<float*>, vector<vector<float>>, 
                        vector<float>, int, int, 
                        int, int);

#endif