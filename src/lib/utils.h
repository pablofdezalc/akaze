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
#include <string.h>
#include <stdint.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>

//*************************************************************************************
//*************************************************************************************

// Declaration of Functions
void compute_min_32F(const cv::Mat& src, float& value);
void compute_max_32F(const cv::Mat& src, float& value);
void convert_scale(cv::Mat& src);
void copy_and_convert_scale(const cv::Mat& src, cv::Mat& dst);
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);
int save_keypoints(const char *outFile, const std::vector<cv::KeyPoint>& kpts,
                   const cv::Mat& desc, const bool& save_desc);
void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         std::vector<cv::Point2f>& pmatches, const float& nndr);
void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
                            std::vector<cv::Point2f>& inliers,
                            const float& error, const bool& use_fund);
void compute_inliers_homography(const std::vector<cv::Point2f>& matches,
                                std::vector<cv::Point2f> &inliers,
                                const cv::Mat&H, const float& min_error);
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs);
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs, const int& color);
void read_homography(const char *hFile, cv::Mat& H1toN);
void show_input_options_help(const int& example);

//*************************************************************************************
//*************************************************************************************

#endif
