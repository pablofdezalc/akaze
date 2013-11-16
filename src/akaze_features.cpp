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
#include "cmdLine.h"

using namespace std;
using namespace cv;

int main( int argc, char *argv[] )
{
  // Variables
  AKAZEOptions options;
  setDefaultAKAZEOptions(options);
  string image_path, key_path;

  // Variable for computation times.
  double t1 = 0.0, t2 = 0.0, tdet = 0.0, tdesc = 0.0;

  // Parse the input command line options
  if(!parse_input_options(options, image_path, key_path, argc, argv))
    return -1;
  if (options.verbosity)
    printAKAZEOptions(options);

  // Try to read the image and if necessary convert to grayscale.
  Mat img = imread(image_path, 0);
  if (img.data == NULL)
  {
    cout << "Error: cannot load image from file:" << endl << image_path << endl;
    return -1;
  }

  // Convert the image to float to extract features.
  Mat img_32;
  img.convertTo(img_32,CV_32F,1.0/255.0,0);
  // Don't forget to specify image dimensions in AKAZE's options.
  options.img_width = img.cols;
  options.img_height = img.rows;

  // Extract features.
  std::vector<cv::KeyPoint> kpts;
  t1 = cv::getTickCount();
  AKAZE evolution(options);
  evolution.Create_Nonlinear_Scale_Space(img_32);
  evolution.Feature_Detection(kpts);
  t2 = cv::getTickCount();
  tdet = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Compute descriptors.
  cv::Mat desc;
  t1 = cv::getTickCount();
  evolution.Compute_Descriptors(kpts,desc);
  t2 = cv::getTickCount();
  tdesc = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Summarize the computation times.
  evolution.Show_Computation_Times();
  evolution.Save_Scale_Space();
  cout << "Number of points: " << kpts.size() << endl;
  cout << "Time Detector: " << tdet << " ms" << endl;
  cout << "Time Descriptor: " << tdesc << " ms" << endl;

  // Save keypoints in ASCII format.
  if (!key_path.empty())
    save_keypoints(&key_path[0],kpts,desc,true);

  // Check out the result visually.
  Mat img_rgb = cv::Mat(Size(img.cols,img.rows),CV_8UC3);
  cvtColor(img,img_rgb,CV_GRAY2BGR);
  draw_keypoints(img_rgb,kpts);
  imshow(image_path, img_rgb);
  waitKey(0);
}

bool parse_input_options(AKAZEOptions& options,
                         std::string& image_filename,
                         std::string& keypoint_filename,
                         int argc, char *argv[])
{
  keypoint_filename.clear();
  image_filename.clear();

  // Create command line options.
  CmdLine cmdLine;
  cmdLine.add(make_switch('h', "help"));
  // Verbose option for debug.
  cmdLine.add(make_option('v', options.verbosity, "verbose"));
  // Image file name.
  cmdLine.add(make_option('i', image_filename, "image"));
  // Scale-space parameters.
  cmdLine.add(make_option('O', options.omax, "omax"));
  cmdLine.add(make_option('S', options.nsublevels, "nsublevels"));
  cmdLine.add(make_option('s', options.soffset, "soffset"));
  cmdLine.add(make_option('d', options.sderivatives, "sderivatives"));
  cmdLine.add(make_option('g', options.diffusivity, "diffusivity"));
  // Detection parameters.
  cmdLine.add(make_option('t', options.dthreshold, "dthreshold"));
  // Descriptor parameters.
  cmdLine.add(make_option('D', options.descriptor, "descriptor"));
  cmdLine.add(make_option('C', options.descriptor_channels, "descriptor_channels"));
  cmdLine.add(make_option('F', options.descriptor_size, "descriptor_size"));
  // Save the keypoints.
  cmdLine.add(make_option('o', keypoint_filename, "output"));
  // Save scale-space
  cmdLine.add(make_option('w', options.save_scale_space, "save_scale_space"));

  // Try to process
  try
  {
    if (argc == 1)
      throw std::string("Invalid command line parameter.");

    cmdLine.process(argc, argv);

    if (!cmdLine.used('i'))
      throw std::string("Invalid command line parameter.");
  }
  catch(const std::string& s)
  {
    show_input_options_help(0);
    return false;
  }

  return true;
}