
#ifndef _UTILS_H_
#define _UTILS_H_

//******************************************************************************
//******************************************************************************

// OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

//******************************************************************************
//******************************************************************************

// Stringify common types such as int, double and others.
template <typename T>
inline std::string to_string(const T& x) {
  std::stringstream oss;
  oss << x;
  return oss.str();
}

//******************************************************************************
//******************************************************************************

// Stringify and format integral types as follows:
// to_formatted_string(  1, 2) produces string:  '01'
// to_formatted_string(  5, 2) produces string:  '05'
// to_formatted_string( 19, 2) produces string:  '19'
// to_formatted_string( 19, 3) produces string: '019'
template <typename Integer>
inline std::string to_formatted_string(Integer x, int num_digits) {
  std::stringstream oss;
  oss << std::setfill('0') << std::setw(num_digits) << x;
  return oss.str();
}

//******************************************************************************
//******************************************************************************

void compute_min_32F(const cv::Mat& src, float& value);
void compute_max_32F(const cv::Mat& src, float& value);
void convert_scale(cv::Mat& src);
void copy_and_convert_scale(const cv::Mat& src, cv::Mat& dst);

void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);
int save_keypoints(const std::string& outFile,
                   const std::vector<cv::KeyPoint>& kpts,
                   const cv::Mat& desc, bool save_desc);

void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         std::vector<cv::Point2f>& pmatches, float nndr);
void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
                            std::vector<cv::Point2f>& inliers,
                            float error, bool use_fund);
void compute_inliers_homography(const std::vector<cv::Point2f>& matches,
                                std::vector<cv::Point2f> &inliers,
                                const cv::Mat&H, float min_error);
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs);
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs, int color);
void read_homography(const std::string& hFile, cv::Mat& H1toN);
void show_input_options_help(int example);

//******************************************************************************
//******************************************************************************

#endif
