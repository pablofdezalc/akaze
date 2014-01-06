/**
 * @file akaze_features.h
 * @brief Main program for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Sep 16, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
// Includes
#include "AKAZE.h"
#include "config.h"

/* ************************************************************************* */
// Declaration of functions
int parse_input_options(AKAZEOptions& options, std::string& img_path,
                        std::string& kpts_path, int argc, char *argv[]);


