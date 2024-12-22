#ifndef UTILS_HOST_H
#define UTILS_HOST_H

#include "libs.h"

void init_weights(vector<vector<float>> &Ws);

void init_param(vector<float> &, vector<float> &,
                vector<float> &, vector<float> &,
                vector<float> &, vector<float> &);

float randValue();

void makeValue(std::vector<float>&, int, int);

#endif