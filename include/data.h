
#ifndef UTILS_DATA_H
#define UTILS_DATA_H

#include "libs.h"

int _reverse_int(int);

void readImages(const string&, vector<vector<float>>&, int&, int&, int&, int&);

void readLabels(const string&, vector<int>& , int& );

#endif
