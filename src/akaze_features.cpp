//=============================================================================
//
// akaze_features.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Georgia Institute of Technology (1)
//               TrueVision Solutions (2)
// Date: 16/09/2013
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2013, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file akaze_features.cpp
 * @brief Main program for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Sep 16, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "akaze_features.h"

// Namespaces
using namespace std;
using namespace cv;

//*************************************************************************************
//*************************************************************************************

/** Main Function 																	 */
int main( int argc, char *argv[] ) {

  // Variables
  AKAZEOptions options;
  Mat img, img_32, img_rgb;
  char img_name[NMAX_CHAR], kfile[NMAX_CHAR];
  double t1 = 0.0, t2 = 0.0, tdet = 0.0, tdesc = 0.0;

  // Parse the input command line options
  if (parse_input_options(options,img_name,kfile,argc,argv)) {
    return -1;
  }

  // Read the image, force to be grey scale
  img = imread(img_name,0);

  if (img.data == NULL) {
    cout << "Error loading image: " << img_name << endl;
    return -1;
  }

  // Convert the image to float
  img.convertTo(img_32,CV_32F,1.0/255.0,0);
  img_rgb = cv::Mat(Size(img.cols,img.rows),CV_8UC3);
  std::vector<cv::KeyPoint> kpts;

  options.img_width = img.cols;
  options.img_height = img.rows;

  // Create the AKAZE object
  AKAZE evolution(options);

  t1 = cv::getTickCount();

  // Create the Gaussian scale space
  evolution.Create_Nonlinear_Scale_Space(img_32);
  evolution.Feature_Detection(kpts);

  t2 = cv::getTickCount();
  tdet = 1000.0*(t2-t1) / cv::getTickFrequency();

  cv::Mat desc;

  // Compute descriptors
  t1 = cv::getTickCount();

  evolution.Compute_Descriptors(kpts,desc);

  t2 = cv::getTickCount();
  tdesc = 1000.0*(t2-t1) / cv::getTickFrequency();

  evolution.Show_Computation_Times();
  evolution.Save_Scale_Space();

  cout << "Number of points: " << kpts.size() << endl;
  cout << "Time Detector: " << tdet << endl;
  cout << "Time Descriptor: " << tdesc << endl;

  // Save keypoints to a txt file
  if (options.save_keypoints == true) {
    save_keypoints(kfile,kpts,desc,true);
  }

  cvtColor(img,img_rgb,CV_GRAY2BGR);
  draw_keypoints(img_rgb,kpts);

  imshow("image.jpg",img_rgb);
  waitKey(0);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * @param options Structure that contains KAZE settings
 * @param img_name Name of the input image
 * @param kfile Name of the file where the keypoints where be stored
 */
int parse_input_options(AKAZEOptions &options, char *img_name, char *kfile, int argc, char *argv[]) {

  // If there is only one argument return
  if (argc == 1) {
    show_input_options_help(0);
    return -1;
  }
  // Set the options from the command line
  else if (argc >= 2)
  {
    options = AKAZEOptions();

    strcpy(kfile,"../output/files/keypoints.txt");

    if( !strcmp(argv[1],"--help") )
    {
      show_input_options_help(0);
      return -1;
    }

    strcpy(img_name,argv[1]);

    for (int i = 2; i < argc; i++) {
      if (!strcmp(argv[i],"--soffset")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.soffset = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--omax")) {
        i = i+1;
        if ( i >= argc ) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.omax = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--dthreshold")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.dthreshold = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--sderivatives")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.sderivatives = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--nsublevels")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.nsublevels = atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--diffusivity")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.diffusivity = atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--descriptor")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.descriptor = atoi(argv[i]);

          if (options.descriptor < 0 || options.descriptor > MLDB) {
            options.descriptor = MLDB;
          }
        }
      }
      else if (!strcmp(argv[i],"--descriptor_channels")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.descriptor_channels = atoi(argv[i]);

          if (options.descriptor_channels <= 0 || options.descriptor_channels > 3) {
            options.descriptor_channels = 3;
          }
        }
      }
      else if (!strcmp(argv[i],"--descriptor_size")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.descriptor_size = atoi(argv[i]);

          if (options.descriptor_size < 0) {
            options.descriptor_size = 0;
          }
        }
      }
      else if (!strcmp(argv[i],"--save_scale_space")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.save_scale_space = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--verbose")) {
        options.verbosity = true;
      }
      else if (!strcmp(argv[i],"--output")) {
        options.save_keypoints = true;
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          strcpy(kfile,argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--help")) {
        // Show the help!!
        show_input_options_help(0);
        return -1;
      }
    }
  }

  return 0;
}
