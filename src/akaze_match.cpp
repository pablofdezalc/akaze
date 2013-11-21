//=============================================================================
//
// akaze_match.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Georgia Institute of Technology (1)
//               TrueVision Solutions (2)
// Date: 15/09/2013
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2013, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file akaze_match.cpp
 * @brief Main program for matching two images with AKAZE features
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla
 *
 * Modification:
 * 16/11/2013: David Ok (david.ok8@gmail.com)
 *
 */

#include "akaze_match.h"
#include "cmdLine.h"

using namespace std;
using namespace cv;

int main( int argc, char *argv[] )
{
  // Variables
  AKAZEOptions options;
  Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
  string img_name1, img_name2, hfile;
  string rfile;
  float ratio = 0.0, rfactor = .60;
  int nkpts1 = 0, nkpts2 = 0, nmatches = 0, ninliers = 0, noutliers = 0;

  vector<KeyPoint> kpts1, kpts2;
  vector<vector<DMatch> > dmatches;
  Mat desc1, desc2;
  Mat HG;

  // Variables for measuring computation times
  double t1 = 0.0, t2 = 0.0;
  double takaze = 0.0, tmatch = 0.0, thomo = 0.0;

  // Parse the input command line options
  if (!parse_input_options(options,img_name1,img_name2,hfile,rfile,argc,argv)) {
    return -1;
  }

  // Read image 1 and if necessary convert to grayscale.
  img1 = imread(img_name1,0);
  if (img1.data == NULL) {
    cout << "Error loading image 1: " << img_name1 << endl;
    return -1;
  }
  // Read image 2 and if necessary convert to grayscale.
  img2 = imread(img_name2,0);
  if (img2.data == NULL) {
    cout << "Error loading image 2: " << img_name2 << endl;
    return -1;
  }
  // Read ground truth homography file
  read_homography(hfile, HG);

  // Convert the images to float
  img1.convertTo(img1_32,CV_32F,1.0/255.0,0);
  img2.convertTo(img2_32,CV_32F,1.0/255.0,0);

  // Color images for results visualization
  img1_rgb = Mat(Size(img1.cols,img1.rows),CV_8UC3);
  img2_rgb = Mat(Size(img2.cols,img1.rows),CV_8UC3);
  img_com = Mat(Size(img1.cols*2,img1.rows),CV_8UC3);
  img_r = Mat(Size(img_com.cols*rfactor,img_com.rows*rfactor),CV_8UC3);

  // Create the first AKAZE object
  options.img_width = img1.cols;
  options.img_height = img1.rows;
  AKAZE evolution1(options);

  // Create the second HKAZE object
  options.img_width = img2.cols;
  options.img_height = img2.rows;
  AKAZE evolution2(options);

  t1 = getTickCount();

  // Create the nonlinear scale space
  // and perform feature detection and description for image 1
  evolution1.Create_Nonlinear_Scale_Space(img1_32);
  evolution1.Feature_Detection(kpts1);
  evolution1.Compute_Descriptors(kpts1,desc1);

  evolution2.Create_Nonlinear_Scale_Space(img2_32);
  evolution2.Feature_Detection(kpts2);
  evolution2.Compute_Descriptors(kpts2,desc2);

  t2 = getTickCount();
  takaze = 1000.0*(t2-t1)/getTickFrequency();

  nkpts1 = kpts1.size();
  nkpts2 = kpts2.size();

  // Matching Descriptors!!
  vector<Point2f> matches, inliers;
  Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
  Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("BruteForce-Hamming");

  t1 = getTickCount();

  if (options.descriptor < MLDB_UPRIGHT) {
    matcher_l2->knnMatch(desc1,desc2,dmatches,2);
  }
  // Binary descriptor, use Hamming distance
  else {
    matcher_l1->knnMatch(desc1,desc2,dmatches,2);
  }

  t2 = cv::getTickCount();
  tmatch = 1000.0*(t2 - t1)/cv::getTickFrequency();

  // Compute Inliers!!
  t1 = getTickCount();

  matches2points_nndr(kpts1,kpts2,dmatches,matches,DRATIO);
  compute_inliers_homography(matches,inliers,HG,MIN_H_ERROR);

  t2 = getTickCount();
  thomo = 1000.0*(t2 - t1)/getTickFrequency();

  // Compute the inliers statistics
  nmatches = matches.size()/2;
  ninliers = inliers.size()/2;
  noutliers = nmatches - ninliers;
  ratio = 100.0*((float) ninliers / (float) nmatches);

  // Prepare the visualization
  cvtColor(img1,img1_rgb,CV_GRAY2BGR);
  cvtColor(img2,img2_rgb,CV_GRAY2BGR);

  // Show matching statistics
  cout << "Number of Keypoints Image 1: " << nkpts1 << endl;
  cout << "Number of Keypoints Image 2: " << nkpts2 << endl;
  cout << "A-KAZE Features Extraction Time (ms): " << takaze << endl;
  cout << "Matching Descriptors Time (ms): " << tmatch << endl;
  cout << "Homography Computation Time (ms): " << thomo << endl;
  cout << "Number of Matches: " << nmatches << endl;
  cout << "Number of Inliers: " << ninliers << endl;
  cout << "Number of Outliers: " << noutliers << endl;
  cout << "Inliers Ratio: " << ratio << endl << endl;

  draw_keypoints(img1_rgb,kpts1);
  draw_keypoints(img2_rgb,kpts2);
  draw_inliers(img1_rgb,img2_rgb,img_com,inliers);
  imshow("Inliers",img_com);
  waitKey(0);
}

bool parse_input_options(AKAZEOptions& options,
                         string& img_name1, string& img_name2,
                         string& hom, string& kfile, int argc, char *argv[])
{
  // Create command line options.
  CmdLine cmdLine;
  cmdLine.add(make_switch('h', "help"));
  // Verbose option for debug.
  cmdLine.add(make_option('v', options.verbosity, "verbose"));
  // Image file name.
  cmdLine.add(make_option('L', img_name1, "left image, i.e. path of image 1"));
  cmdLine.add(make_option('R', img_name2, "right image, i.e. path of image 2"));
  cmdLine.add(make_option('H', hom, "ground truth homography"));
  // Scale-space parameters.
  cmdLine.add(make_option('O', options.omax, "omax"));
  cmdLine.add(make_option('S', options.nsublevels, "nsublevels"));
  cmdLine.add(make_option('s', options.soffset, "soffset"));
  cmdLine.add(make_option('d', options.sderivatives, "sderivatives"));
  cmdLine.add(make_option('g', options.diffusivity, "diffusivity"));
  // Detection parameters.
  cmdLine.add(make_option('a', options.dthreshold, "dthreshold"));
  cmdLine.add(make_option('b', options.dthreshold2, "dthreshold2")); // ?????
  // Descriptor parameters.
  cmdLine.add(make_option('D', options.descriptor, "descriptor"));
  cmdLine.add(make_option('C', options.descriptor_channels, "descriptor_channels"));
  cmdLine.add(make_option('F', options.descriptor_size, "descriptor_size"));
  // Save the keypoints.
  cmdLine.add(make_option('o', kfile, "output"));
  // Save scale-space
  cmdLine.add(make_option('w', options.save_scale_space, "save_scale_space"));

  // Try to process
  try
  {
    if (argc == 1)
      throw std::string("Invalid command line parameter.");

    cmdLine.process(argc, argv);

    if (!cmdLine.used('L') || !cmdLine.used('R') || !cmdLine.used('H'))
      throw std::string("Invalid command line parameter.");
  }
  catch(const std::string& s)
  {
    show_input_options_help(1);
    return false;
  }

  return true;
}