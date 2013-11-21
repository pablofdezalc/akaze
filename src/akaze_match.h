
#ifndef _AKAZE_MATCH_H_
#define _AKAZE_MATCH_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include "AKAZE.h"
#include "config.h"
#include "utils.h"

//*************************************************************************************
//*************************************************************************************

// Image matching options
const bool COMPUTE_INLIERS_RANSAC = true;	// 0->Use ground truth homography, 1->Estimate homography with RANSAC
const float MIN_H_ERROR = 2.50f;	// Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;		// NNDR Matching value

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
int parse_input_options(AKAZEOptions &options, std::string& img_path1,
                        std::string& img_path2, std::string& homography_path,
                        int argc, char *argv[]);

//*************************************************************************************
//*************************************************************************************

#endif
