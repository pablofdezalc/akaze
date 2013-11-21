#ifndef AKAZE_COMPARE_H
#define AKAZE_COMPARE_H

#include "AKAZE.h"
#include "config.h"

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
const bool COMPUTE_HOMOGRAPHY = false;	// false -> Use ground truth homography
                                        // true  -> Estimate homography with RANSAC
const float MIN_H_ERROR = 2.50f;	      // Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;		          // NNDR Matching value


/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * and image matching between two input images
 * @param options Structure that contains KAZE settings
 * @param img_name1 Name of the first input image
 * @param img_name2 Name of the second input image
 * @param hom Name of the file that contains a ground truth homography
 * @param kfile Name of the file where the keypoints where be stored
 */
bool parse_input_options(AKAZEOptions &options,
                         std::string& img_name1, std::string& img_name2,
                         std::string& hom, std::string& kfile,
                         int argc, char *argv[]);

#endif // AKAZE_COMPARE_H
