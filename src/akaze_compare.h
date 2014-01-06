
/**
 * @file akaze_compare.h
 * @brief Main program for matching two images with A-KAZE features and compare
 * to BRISK and ORB
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla
 */

#pragma once

/* ************************************************************************* */
// Includes
#include "AKAZE.h"
#include "config.h"

/* ************************************************************************* */
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

// Some image matching options
const bool COMPUTE_INLIERS_RANSAC = false;	// 0->Use ground truth homography, 1->Estimate homography with RANSAC
const float MIN_H_ERROR = 2.50f;	      // Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;		          // NNDR Matching value

/* ************************************************************************* */
// Declaration of functions
int parse_input_options(AKAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]);

