#ifndef FED_H
#define FED_H

//******************************************************************************
//******************************************************************************

// Includes
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <vector>

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
int fed_tau_by_process_time(float T, int M, float tau_max, bool reordering, std::vector<float> &tau);
int fed_tau_by_cycle_time(float t, float tau_max, bool reordering, std::vector<float> &tau);
int fed_tau_internal(int n, float scale, float tau_max, bool reordering, std::vector<float> &tau);
bool fed_is_prime_internal(int number);

//*************************************************************************************
//*************************************************************************************

#endif // FED_H
