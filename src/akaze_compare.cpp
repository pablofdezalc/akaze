//=============================================================================
//
// akaze_compare.cpp
// Authors: Pablo F. Alcantarilla, Jesus Nuevo (2)
// Institutions: TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file akaze_compare.cpp
 * @brief Main program for matching two images with A-KAZE features and compare
 * to BRISK and ORB
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla
 */

#include "./lib/AKAZE.h"

// OpenCV
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

/* ************************************************************************* */
// ORB settings
int ORB_MAX_KPTS = 1500;
float ORB_SCALE_FACTOR = 1.2;
int ORB_PYRAMID_LEVELS = 4;
float ORB_EDGE_THRESHOLD = 31.0;
int ORB_FIRST_PYRAMID_LEVEL = 0;
int ORB_WTA_K = 2;
int ORB_PATCH_SIZE = 31;

// BRISK settings
int BRISK_HTHRES = 10;
int BRISK_NOCTAVES = 4;

// Some image matching options
float MIN_H_ERROR = 2.50f;	      // Maximum error in pixels to accept an inlier
float DRATIO = 0.80f;		          // NNDR Matching value

/* ************************************************************************* */
/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * and image matching between two input images
 * @param options Structure that contains KAZE settings
 * @param img_path1 Path for the first input image
 * @param img_path2 Path for the second input image
 * @param homography_path Path for the file that contains a ground truth homography
 */
int parse_input_options(AKAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]);

/* ************************************************************************* */
int main(int argc, char *argv[]) {

  // Variables
  AKAZEOptions options;
  cv::Mat img1, img1_32, img2, img2_32;
  string img_path1, img_path2, homography_path;
  double t1 = 0.0, t2 = 0.0;

  // ORB variables
  vector<cv::KeyPoint> kpts1_orb, kpts2_orb;
  vector<cv::Point2f> matches_orb, inliers_orb;
  vector<vector<cv::DMatch> > dmatches_orb;
  cv::Mat desc1_orb, desc2_orb;
  int nmatches_orb = 0, ninliers_orb = 0, noutliers_orb = 0;
  int nkpts1_orb = 0, nkpts2_orb = 0;
  float ratio_orb = 0.0;
  double torb = 0.0;

  // BRISK variables
  vector<cv::KeyPoint> kpts1_brisk, kpts2_brisk;
  vector<cv::Point2f> matches_brisk, inliers_brisk;
  vector<vector<cv::DMatch> > dmatches_brisk;
  cv::Mat desc1_brisk, desc2_brisk;
  int nmatches_brisk = 0, ninliers_brisk = 0, noutliers_brisk = 0;
  int nkpts1_brisk = 0, nkpts2_brisk = 0;
  float ratio_brisk = 0.0;
  double tbrisk = 0.0;

  // AKAZE variables
  vector<cv::KeyPoint> kpts1_akaze, kpts2_akaze;
  vector<cv::Point2f> matches_akaze, inliers_akaze;
  vector<vector<cv::DMatch> > dmatches_akaze;
  cv::Mat desc1_akaze, desc2_akaze;
  int nmatches_akaze = 0, ninliers_akaze = 0, noutliers_akaze = 0;
  int nkpts1_akaze = 0, nkpts2_akaze = 0;
  float ratio_akaze = 0.0;
  double takaze = 0.0;

  // Create the L2 and L1 matchers
  cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce");
  cv::Ptr<cv::DescriptorMatcher> matcher_l1 = cv::DescriptorMatcher::create("BruteForce-Hamming");
  cv::Mat HG;

  // Parse the input command line options
  if (parse_input_options(options,img_path1,img_path2,homography_path,argc,argv))
    return -1;

  // Read the image, force to be grey scale
  img1 = cv::imread(img_path1,0);

  if (img1.data == NULL) {
    cerr << "Error loading image: " << img_path1 << endl;
    return -1;
  }

  // Read the image, force to be grey scale
  img2 = cv::imread(img_path2,0);

  if (img2.data == NULL) {
    cerr << "Error loading image: " << img_path2 << endl;
    return -1;
  }

  // Convert the images to float
  img1.convertTo(img1_32,CV_32F,1.0/255.0,0);
  img2.convertTo(img2_32,CV_32F,1.0/255.0,0);

  // Color images for results visualization
  cv::Mat img1_rgb_orb = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
  cv::Mat img2_rgb_orb = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
  cv::Mat img_com_orb = cv::Mat(cv::Size(img1.cols*2, img1.rows), CV_8UC3);

  cv::Mat img1_rgb_brisk = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
  cv::Mat img2_rgb_brisk = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
  cv::Mat img_com_brisk = cv::Mat(cv::Size(img1.cols*2, img1.rows), CV_8UC3);

  cv::Mat img1_rgb_akaze = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
  cv::Mat img2_rgb_akaze = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
  cv::Mat img_com_akaze = cv::Mat(cv::Size(img1.cols*2, img1.rows), CV_8UC3);

  // Read the homography file
  bool use_ransac = false;
  if (read_homography(homography_path,HG) == false)
    use_ransac = true;

  /* ************************************************************************* */
  // ORB Features
  //*****************
#if CV_VERSION_EPOCH == 2
  cv::ORB orb(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
#else
  cv::Ptr<cv::ORB> orb = cv::ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
#endif
    ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, cv::ORB::HARRIS_SCORE, ORB_PATCH_SIZE);

  t1 = cv::getTickCount();

#if CV_VERSION_EPOCH == 2
  orb(img1, cv::noArray(), kpts1_orb, desc1_orb, false);
  orb(img2, cv::noArray(), kpts2_orb, desc2_orb, false);
#else
  orb->detectAndCompute(img1, cv::noArray(), kpts1_orb, desc1_orb, false);
  orb->detectAndCompute(img2, cv::noArray(), kpts2_orb, desc2_orb, false);
#endif

  matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);
  matches2points_nndr(kpts1_orb,kpts2_orb,dmatches_orb,matches_orb,DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(matches_orb,inliers_orb, HG, MIN_H_ERROR);
  else
    compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);

  nkpts1_orb = kpts1_orb.size();
  nkpts2_orb = kpts2_orb.size();
  nmatches_orb = matches_orb.size()/2;
  ninliers_orb = inliers_orb.size()/2;
  noutliers_orb = nmatches_orb-ninliers_orb;
  ratio_orb = 100.0*(float)(ninliers_orb)/(float)(nmatches_orb);

  t2 = cv::getTickCount();
  torb = 1000.0*(t2-t1) / cv::getTickFrequency();

  if (options.show_results == true) {
    cvtColor(img1, img1_rgb_orb, cv::COLOR_GRAY2BGR);
    cvtColor(img2, img2_rgb_orb, cv::COLOR_GRAY2BGR);

    draw_keypoints(img1_rgb_orb, kpts1_orb);
    draw_keypoints(img2_rgb_orb, kpts2_orb);
    draw_inliers(img1_rgb_orb, img2_rgb_orb, img_com_orb, inliers_orb, 0);

    cout << "ORB Results" << endl;
    cout << "**************************************" << endl;
    cout << "Number of Keypoints Image 1: " << nkpts1_orb << endl;
    cout << "Number of Keypoints Image 2: " << nkpts2_orb << endl;
    cout << "Number of Matches: " << nmatches_orb << endl;
    cout << "Number of Inliers: " << ninliers_orb << endl;
    cout << "Number of Outliers: " << noutliers_orb << endl;
    cout << "Inliers Ratio: " << ratio_orb << endl;
    cout << "ORB Features Extraction Time (ms): " << torb << endl;
    cout << endl;
  }

  /* ************************************************************************* */
  // BRISK Features
  //*****************
#if CV_VERSION_EPOCH == 2
  cv::BRISK brisk(BRISK_HTHRES, BRISK_NOCTAVES, 1.0f);
#else
  cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(BRISK_HTHRES, BRISK_NOCTAVES, 1.0f);
#endif

  t1 = cv::getTickCount();

#if CV_VERSION_EPOCH == 2
  brisk(img1, cv::noArray(), kpts1_brisk, desc1_brisk, false);
  brisk(img2, cv::noArray(), kpts2_brisk, desc2_brisk, false);
#else
  brisk->detectAndCompute(img1, cv::noArray(), kpts1_brisk, desc1_brisk, false);
  brisk->detectAndCompute(img2, cv::noArray(), kpts2_brisk, desc2_brisk, false);
#endif

  matcher_l1->knnMatch(desc1_brisk, desc2_brisk, dmatches_brisk, 2);
  matches2points_nndr(kpts1_brisk, kpts2_brisk, dmatches_brisk, matches_brisk, DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(matches_brisk, inliers_brisk, HG, MIN_H_ERROR);
  else
    compute_inliers_ransac(matches_brisk, inliers_brisk, MIN_H_ERROR, false);

  nkpts1_brisk = kpts1_brisk.size();
  nkpts2_brisk= kpts2_brisk.size();
  nmatches_brisk = matches_brisk.size()/2;
  ninliers_brisk = inliers_brisk.size()/2;
  noutliers_brisk = nmatches_brisk-ninliers_brisk;
  ratio_brisk = 100.0*(float)(ninliers_brisk)/(float)(nmatches_brisk);

  t2 = cv::getTickCount();
  tbrisk = 1000.0*(t2-t1) / cv::getTickFrequency();

  if (options.show_results == true) {
    cvtColor(img1, img1_rgb_brisk, cv::COLOR_GRAY2BGR);
    cvtColor(img2, img2_rgb_brisk, cv::COLOR_GRAY2BGR);

    draw_keypoints(img1_rgb_brisk, kpts1_brisk);
    draw_keypoints(img2_rgb_brisk, kpts2_brisk);
    draw_inliers(img1_rgb_brisk, img2_rgb_brisk, img_com_brisk, inliers_brisk, 1);

    cout << "BRISK Results" << endl;
    cout << "**************************************" << endl;
    cout << "Number of Keypoints Image 1: " << nkpts1_brisk << endl;
    cout << "Number of Keypoints Image 2: " << nkpts2_brisk << endl;
    cout << "Number of Matches: " << nmatches_brisk << endl;
    cout << "Number of Inliers: " << ninliers_brisk << endl;
    cout << "Number of Outliers: " << noutliers_brisk << endl;
    cout << "Inliers Ratio: " << ratio_brisk << endl;
    cout << "BRISK Features Extraction Time (ms): " << tbrisk << endl;
    cout << endl;
  }

  /* ************************************************************************* */
  // A-KAZE Features
  //*******************
  options.img_width = img1.cols;
  options.img_height = img1.rows;
  libAKAZE::AKAZE evolution1(options);

  options.img_width = img2.cols;
  options.img_height = img2.rows;
  libAKAZE::AKAZE evolution2(options);

  t1 = cv::getTickCount();

  evolution1.Create_Nonlinear_Scale_Space(img1_32);
  evolution1.Feature_Detection(kpts1_akaze);
  evolution1.Compute_Descriptors(kpts1_akaze, desc1_akaze);

  evolution2.Create_Nonlinear_Scale_Space(img2_32);
  evolution2.Feature_Detection(kpts2_akaze);
  evolution2.Compute_Descriptors(kpts2_akaze, desc2_akaze);

  if (options.descriptor < MLDB_UPRIGHT)
    matcher_l2->knnMatch(desc1_akaze, desc2_akaze, dmatches_akaze, 2);

  // Binary descriptor, use Hamming distance
  else
    matcher_l1->knnMatch(desc1_akaze, desc2_akaze, dmatches_akaze, 2);

  matches2points_nndr(kpts1_akaze,kpts2_akaze,dmatches_akaze,matches_akaze,DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(matches_akaze,inliers_akaze,HG,MIN_H_ERROR);
  else
    compute_inliers_ransac(matches_akaze,inliers_akaze,MIN_H_ERROR,false);

  t2 = cv::getTickCount();
  takaze = 1000.0*(t2-t1)/cv::getTickFrequency();

  nkpts1_akaze = kpts1_akaze.size();
  nkpts2_akaze = kpts2_akaze.size();
  nmatches_akaze = matches_akaze.size()/2;
  ninliers_akaze = inliers_akaze.size()/2;
  noutliers_akaze = nmatches_akaze-ninliers_akaze;
  ratio_akaze = 100.0*((float) ninliers_akaze / (float) nmatches_akaze);

  if (options.show_results == true) {
    cvtColor(img1,img1_rgb_akaze, cv::COLOR_GRAY2BGR);
    cvtColor(img2,img2_rgb_akaze, cv::COLOR_GRAY2BGR);

    draw_keypoints(img1_rgb_akaze, kpts1_akaze);
    draw_keypoints(img2_rgb_akaze, kpts2_akaze);
    draw_inliers(img1_rgb_akaze, img2_rgb_akaze, img_com_akaze, inliers_akaze, 2);

    cout << "A-KAZE Results" << endl;
    cout << "**************************************" << endl;
    cout << "Number of Keypoints Image 1: " << nkpts1_akaze << endl;
    cout << "Number of Keypoints Image 2: " << nkpts2_akaze << endl;
    cout << "Number of Matches: " << nmatches_akaze << endl;
    cout << "Number of Inliers: " << ninliers_akaze << endl;
    cout << "Number of Outliers: " << noutliers_akaze << endl;
    cout << "Inliers Ratio: " << ratio_akaze << endl;
    cout << "A-KAZE Features Extraction Time (ms): " << takaze << endl;
    cout << endl;

    // Show the images with the inliers
    cv::imshow("ORB",img_com_orb);
    cv::imshow("BRISK",img_com_brisk);
    cv::imshow("A-KAZE",img_com_akaze);
    cv::waitKey(0);
  }
}

/* ************************************************************************* */
int parse_input_options(AKAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]) {

  // If there is only one argument return
  if (argc == 1) {
    show_input_options_help(2);
    return -1;
  }
  // Set the options from the command line
  else if (argc >= 2) {
    options = AKAZEOptions();

    if (!strcmp(argv[1],"--help")) {
      show_input_options_help(2);
      return -1;
    }

    img_path1 = argv[1];
    img_path2 = argv[2];

    if (argc >= 4)
      homography_path = argv[3];

    for (int i = 3; i < argc; i++) {
      if (!strcmp(argv[i],"--soffset")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.soffset = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--omax")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.omax = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--dthreshold")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.dthreshold = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--sderivatives")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.sderivatives = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--nsublevels")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.nsublevels = atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--diffusivity")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.diffusivity = DIFFUSIVITY_TYPE(atoi(argv[i]));
        }
      }
      else if (!strcmp(argv[i],"--descriptor")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.descriptor = DESCRIPTOR_TYPE(atoi(argv[i]));

          if (options.descriptor < 0 || options.descriptor > MLDB) {
            options.descriptor = MLDB;
          }
        }
      }
      else if (!strcmp(argv[i],"--descriptor_channels")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
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
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.descriptor_size = atoi(argv[i]);

          if (options.descriptor_size < 0) {
            options.descriptor_size = 0;
          }
        }
      }
      else if (!strcmp(argv[i],"--show_results")) {
        i = i+1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.show_results = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--verbose")) {
        options.verbosity = true;
      }
      else if (!strcmp(argv[i],"--help")) {
        // Show the help!!
        show_input_options_help(2);
        return -1;
      }
      else if (!strncmp(argv[i],"--",2))
        cerr << "Unknown command "<<argv[i] << endl;
    }
  }
  else {
    cerr << "Error introducing input options!!" << endl;
    show_input_options_help(2);
    return -1;
  }

  return 0;
}

