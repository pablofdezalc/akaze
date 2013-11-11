#ifndef AKAZE_COMPARE_H
#define AKAZE_COMPARE_H

//*************************************************************************************
//*************************************************************************************

// Includes
#include "AKAZE.h"
#include "config.h"

//*************************************************************************************
//*************************************************************************************

// ORB settings
const int ORB_MAX_KPTS = 1500;
const float ORB_SCALE_FACTOR = 1.5;
const int ORB_PYRAMID_LEVELS = 3;
const float ORB_EDGE_THRESHOLD = 31.0;
const int ORB_FIRST_PYRAMID_LEVEL = 0;
const int ORB_WTA_K = 2;
const int ORB_PATCH_SIZE = 31;

// BRISK settings
const float BRISK_HTHRES = 10.0;
const int BRISK_NOCTAVES = 3;

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
int parse_input_options(AKAZEOptions &options, char *img_name1, char *img_name2, char *hom,
                        char *kfile, int argc, char *argv[]);

//*************************************************************************************
//*************************************************************************************

#endif // AKAZE_COMPARE_H
