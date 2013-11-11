#ifndef _NLDIFFUSION_FUNCTIONS_H_
#define _NLDIFFUSION_FUNCTIONS_H_

//******************************************************************************
//******************************************************************************

// Includes
#include <opencv2/opencv.hpp>

// OpenMP Includes
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
void Gaussian_2D_Convolution(const cv::Mat &src, cv::Mat &dst, size_t ksize_x, size_t ksize_y, const float sigma);
void Image_Derivatives_Scharr(const cv::Mat &src, cv::Mat &dst, size_t xorder, size_t yorder);
void PM_G1(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, const float k);
void PM_G2(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, const float k);
void Weickert_Diffusivity(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, const float k);
float Compute_K_Percentile(const cv::Mat &img, const float& perc, const float& gscale,
                           const size_t& nbins, const size_t& ksize_x, const size_t& ksize_y);
void Compute_Scharr_Derivatives(const cv::Mat &src, cv::Mat &dst, const int& xorder,
                                const int& yorder, const int& scale);
void NLD_Step_Scalar(cv::Mat &Lt, const cv::Mat &c, cv::Mat &Lstep, float stepsize);
void Downsample_Image(const cv::Mat &src, cv::Mat &dst);
void Halfsample_Image(const cv::Mat &src, cv::Mat &dst);
void Compute_Deriv_Kernels(cv::OutputArray &kx_, cv::OutputArray &ky_, const int& dx, const int& dy, const int& scale);

//*************************************************************************************
//*************************************************************************************


#endif
