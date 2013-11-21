#ifndef _AKAZE_MATCH_H_
#define _AKAZE_MATCH_H_

#include "AKAZE.h"
#include "config.h"
#include "utils.h"

// Image matching parameters
const bool COMPUTE_HOMOGRAPHY = false; // false: Use ground truth homography
                                       // true: Estimate homography with RANSAC
const float MIN_H_ERROR = 2.50f;       // Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;		         // NNDR Matching value

bool parse_input_options(AKAZEOptions &options,
                         std::string& img_name1, std::string& img_name2,
                         std::string& hom, std::string& kfile,
                         int argc, char *argv[]);

#endif