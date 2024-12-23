#ifndef UTILS_HOST_H
#define UTILS_HOST_H

#include "libs.h"

float _randValue();

void _makeValue(std::vector<float>&, int, int);

void init_weights(vector<vector<float>> &Ws);

void init_param(vector<float> &, vector<float> &,
                vector<float> &, vector<float> &,
                vector<float> &, vector<float> &);

float* _transpose(float*, int, int);

float* _add_CPU(float*, float*, int, float);

float* _ewmul_CPU(float*, float*, int);

float* _matmul_CPU(float*, float*, int, int, int);

float* _dReLU_CPU(float*, int);

float _sum_CPU(float*, int);

// vector<float*> _fw_CPU(vector<float>, vector<vector<float>>, int, int, 
//                         int, int);


#endif