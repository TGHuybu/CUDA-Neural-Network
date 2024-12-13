#ifndef UTILS_HOST_H
#define UTILS_HOST_H
#include "libs.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

int ReverseInt(int);
void readImages(const string&, vector<vector<float>>&, int&, int&, int&, int&);
void saveImageAsPNG(const vector<float>&, const string&, int, int);

#endif
