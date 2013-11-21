#ifndef _AKAZE_FEATURES_H_
#define _AKAZE_FEATURES_H_

#include "AKAZE.h"
#include "config.h"

// Declaration of functions
bool parse_input_options(AKAZEOptions& options,
                         std::string& image_filename,
                         std::string& keypoint_filename,
                         int argc, char *argv[]);

#endif