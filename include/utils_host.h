#ifndef UTILS_HOST_H
#define UTILS_HOST_H
#include "libs.h"

int ReverseInt(int);
void readImages(const string&, vector<vector<float>>&, int&, int&, int&, int&);
//void saveImageAsPNG(const vector<float>&, const string&, int, int);
void readLabels(const string&, vector<int>& , int& );
void init_param(vector<float> &, vector<float> &,
                vector<float> &, vector<float> &,
                vector<float> &, vector<float> &);
float randValue();
void makeValue(std::vector<float>&, int, int);
#endif
