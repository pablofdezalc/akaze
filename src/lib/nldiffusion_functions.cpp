//=============================================================================
//
// nldiffusion_functions.cpp
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
 * @file nldiffusion_functions.cpp
 * @brief Functions for nonlinear diffusion filtering applications
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "nldiffusion_functions.h"

using namespace std;
using namespace cv;

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function smoothes an image with a Gaussian kernel
 * @param src Input image
 * @param dst Output image
 * @param ksize_x Kernel size in X-direction (horizontal)
 * @param ksize_y Kernel size in Y-direction (vertical)
 * @param sigma Kernel standard deviation
 */
void Gaussian_2D_Convolution(const cv::Mat &src, cv::Mat &dst, size_t ksize_x, size_t ksize_y, const float sigma) {
  // Compute an appropriate kernel size according to the specified sigma
  if (sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0) {
    ksize_x = ceil(2.0*(1.0 + (sigma-0.8)/(0.3)));
    ksize_y = ksize_x;
  }

  // The kernel size must be and odd number
  if ((ksize_x % 2) == 0) {
    ksize_x += 1;
  }

  if ((ksize_y % 2) == 0) {
    ksize_y += 1;
  }

  // Perform the Gaussian Smoothing with border replication
  GaussianBlur(src,dst,cv::Size(ksize_x,ksize_y),sigma,sigma,cv::BORDER_REPLICATE);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes image derivatives with Scharr kernel
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 * @note Scharr operator approximates better rotation invariance than
 * other stencils such as Sobel. See Weickert and Scharr,
 * A Scheme for Coherence-Enhancing Diffusion Filtering with Optimized Rotation Invariance,
 * Journal of Visual Communication and Image Representation 2002
 */
void Image_Derivatives_Scharr(const cv::Mat &src, cv::Mat &dst, size_t xorder, size_t yorder) {
  Scharr(src,dst,CV_32F,xorder,yorder,1.0,0,cv::BORDER_DEFAULT);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the Perona and Malik conductivity coefficient g1
 * g1 = exp(-|dL|^2/k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void PM_G1(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, const float k)
{
  exp(-(Lx.mul(Lx)+Ly.mul(Ly))/(k*k),dst);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the Perona and Malik conductivity coefficient g2
 * g2 = 1 / (1 + dL^2 / k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void PM_G2(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, const float k) {
  dst = 1.0/(1.0+(Lx.mul(Lx)+Ly.mul(Ly))/(k*k));
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes Weickert conductivity coefficient g3
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 * @note For more information check the following paper: J. Weickert
 * Applications of nonlinear diffusion in image processing and computer vision,
 * Proceedings of Algorithmy 2000
 */
void Weickert_Diffusivity(const cv::Mat &Lx, const cv::Mat &Ly, cv::Mat &dst, const float k) {
  Mat modg;
  pow((Lx.mul(Lx) + Ly.mul(Ly))/(k*k),4,modg);
  cv::exp(-3.315/modg, dst);
  dst = 1.0 - dst;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes a good empirical value for the k contrast factor
 * given an input image, the percentile (0-1), the gradient scale and the number of
 * bins in the histogram
 * @param img Input image
 * @param perc Percentile of the image gradient histogram (0-1)
 * @param gscale Scale for computing the image gradient histogram
 * @param nbins Number of histogram bins
 * @param ksize_x Kernel size in X-direction (horizontal) for the Gaussian smoothing kernel
 * @param ksize_y Kernel size in Y-direction (vertical) for the Gaussian smoothing kernel
 * @return k contrast factor
 */
float Compute_K_Percentile(const cv::Mat &img, const float& perc, const float& gscale,
                           const size_t& nbins, const size_t& ksize_x, const size_t& ksize_y) {
  size_t nbin = 0, nelements = 0, nthreshold = 0, k = 0;
  float kperc = 0.0, modg = 0.0, lx = 0.0, ly = 0.0;
  float npoints = 0.0;
  float hmax = 0.0;

  // Create the array for the histogram
  float *hist = new float[nbins];

  // Create the matrices
  Mat gaussian = cv::Mat::zeros(img.rows,img.cols,CV_32F);
  Mat Lx = cv::Mat::zeros(img.rows,img.cols,CV_32F);
  Mat Ly = cv::Mat::zeros(img.rows,img.cols,CV_32F);

  // Set the histogram to zero, just in case
  for (size_t i = 0; i < nbins; i++) {
    hist[i] = 0.0;
  }

  // Perform the Gaussian convolution
  Gaussian_2D_Convolution(img,gaussian,ksize_x,ksize_y,gscale);

  // Compute the Gaussian derivatives Lx and Ly
  Image_Derivatives_Scharr(gaussian,Lx,1,0);
  Image_Derivatives_Scharr(gaussian,Ly,0,1);

  // Skip the borders for computing the histogram
  for (int i = 1; i < gaussian.rows-1; i++) {
    for (int j = 1; j < gaussian.cols-1; j++) {
      lx = *(Lx.ptr<float>(i)+j);
      ly = *(Ly.ptr<float>(i)+j);
      modg = sqrt(lx*lx + ly*ly);

      // Get the maximum
      if (modg > hmax) {
        hmax = modg;
      }
    }
  }

  // Skip the borders for computing the histogram
  for (int i = 1; i < gaussian.rows-1; i++) {
    for (int j = 1; j < gaussian.cols-1; j++) {
      lx = *(Lx.ptr<float>(i)+j);
      ly = *(Ly.ptr<float>(i)+j);
      modg = sqrt(lx*lx + ly*ly);

      // Find the correspondent bin
      if (modg != 0.0) {
        nbin = floor(nbins*(modg/hmax));

        if (nbin == nbins) {
          nbin--;
        }

        hist[nbin]++;
        npoints++;
      }
    }
  }

  // Now find the perc of the histogram percentile
  nthreshold = (size_t)(npoints*perc);

  for (k = 0; nelements < nthreshold && k < nbins; k++) {
    nelements = nelements + hist[k];
  }

  if (nelements < nthreshold) {
    kperc = 0.03;
  }
  else {
    kperc = hmax*((float)(k)/(float)nbins);
  }

  delete [] hist;
  return kperc;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes Scharr image derivatives
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 * @param scale Scale factor for the derivative size
 */
void Compute_Scharr_Derivatives(const cv::Mat &src, cv::Mat &dst, const int& xorder,
                                const int& yorder, const int& scale) {
  Mat kx, ky;
  Compute_Deriv_Kernels(kx, ky, xorder,yorder,scale);
  sepFilter2D(src,dst,CV_32F,kx,ky);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function performs a scalar non-linear diffusion step
 * @param Ld2 Output image in the evolution
 * @param c Conductivity image
 * @param Lstep Previous image in the evolution
 * @param stepsize The step size in time units
 * @note Forward Euler Scheme 3x3 stencil
 * The function c is a scalar value that depends on the gradient norm
 * dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
 */
void NLD_Step_Scalar(cv::Mat &Ld, const cv::Mat &c, cv::Mat &Lstep, float stepsize) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 1; i < Lstep.rows-1; i++) {
    for (int j = 1; j < Lstep.cols-1; j++) {
      float xpos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(i)+j+1)))*((*(Ld.ptr<float>(i)+j+1))-(*(Ld.ptr<float>(i)+j)));
      float xneg = ((*(c.ptr<float>(i)+j-1))+(*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j))-(*(Ld.ptr<float>(i)+j-1)));

      float ypos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(i+1)+j)))*((*(Ld.ptr<float>(i+1)+j))-(*(Ld.ptr<float>(i)+j)));
      float yneg = ((*(c.ptr<float>(i-1)+j))+(*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j))-(*(Ld.ptr<float>(i-1)+j)));

      *(Lstep.ptr<float>(i)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
    }
  }

  for (int j = 1; j < Lstep.cols-1; j++) {
    float xpos = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(0)+j+1)))*((*(Ld.ptr<float>(0)+j+1))-(*(Ld.ptr<float>(0)+j)));
    float xneg = ((*(c.ptr<float>(0)+j-1))+(*(c.ptr<float>(0)+j)))*((*(Ld.ptr<float>(0)+j))-(*(Ld.ptr<float>(0)+j-1)));

    float ypos = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(1)+j)))*((*(Ld.ptr<float>(1)+j))-(*(Ld.ptr<float>(0)+j)));
    float yneg = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(0)+j)))*((*(Ld.ptr<float>(0)+j))-(*(Ld.ptr<float>(0)+j)));

    *(Lstep.ptr<float>(0)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  for (int j = 1; j < Lstep.cols-1; j++) {
    float xpos = ((*(c.ptr<float>(Lstep.rows-1)+j))+(*(c.ptr<float>(Lstep.rows-1)+j+1)))*((*(Ld.ptr<float>(Lstep.rows-1)+j+1))-(*(Ld.ptr<float>(Lstep.rows-1)+j)));
    float xneg = ((*(c.ptr<float>(Lstep.rows-1)+j-1))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-1)+j-1)));

    float ypos = ((*(c.ptr<float>(Lstep.rows-1)+j))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-1)+j)));
    float yneg = ((*(c.ptr<float>(Lstep.rows-2)+j))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-2)+j)));

    *(Lstep.ptr<float>(Lstep.rows-1)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  for (int i = 1; i < Lstep.rows-1; i++) {
    float xpos = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i)+1)))*((*(Ld.ptr<float>(i)+1))-(*(Ld.ptr<float>(i))));
    float xneg = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i)))-(*(Ld.ptr<float>(i))));

    float ypos = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i+1))))*((*(Ld.ptr<float>(i+1)))-(*(Ld.ptr<float>(i))));
    float yneg = ((*(c.ptr<float>(i-1)))+(*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i)))-(*(Ld.ptr<float>(i-1))));

    *(Lstep.ptr<float>(i)) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  for (int i = 1; i < Lstep.rows-1; i++) {
    float xpos = ((*(c.ptr<float>(i)+Lstep.cols-1))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-1)));
    float xneg = ((*(c.ptr<float>(i)+Lstep.cols-2))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-2)));

    float ypos = ((*(c.ptr<float>(i)+Lstep.cols-1))+(*(c.ptr<float>(i+1)+Lstep.cols-1)))*((*(Ld.ptr<float>(i+1)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-1)));
    float yneg = ((*(c.ptr<float>(i-1)+Lstep.cols-1))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i-1)+Lstep.cols-1)));

    *(Lstep.ptr<float>(i)+Lstep.cols-1) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
  }

  Ld = Ld + Lstep;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function downsamples the input image with the kernel [1/4,1/2,1/4]
 * @param img Input image to be downsampled
 * @param dst Output image with half of the resolution of the input image
 */
void Downsample_Image(const cv::Mat &src, cv::Mat &dst)
{
  int i1 = 0, j1 = 0, i2 = 0, j2 = 0;

  for (i1 = 1; i1 < src.rows; i1+=2) {
    j2 = 0;
    for (j1 = 1; j1 < src.cols; j1+=2) {
      *(dst.ptr<float>(i2)+j2) = 0.5*(*(src.ptr<float>(i1)+j1))+0.25*(*(src.ptr<float>(i1)+j1-1) + *(src.ptr<float>(i1)+j1+1));
      j2++;
    }

    i2++;
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function downsamples the input image using OpenCV resize
 * @param img Input image to be downsampled
 * @param dst Output image with half of the resolution of the input image
 */
void Halfsample_Image(const cv::Mat &src, cv::Mat &dst) {
  // Make sure the destination image is of the right size
  assert(src.cols/2==dst.cols);
  assert(src.rows/2==dst.rows);
  cv::resize(src,dst,dst.size(),0,0,cv::INTER_AREA);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief Compute Scharr derivative kernels for sizes different than 3
 * @param kx_ The derivative kernel in x-direction
 * @param ky_ The derivative kernel in y-direction
 * @param dx The derivative order in x-direction
 * @param dy The derivative order in y-direction
 * @param scale The kernel size
 */
void Compute_Deriv_Kernels(cv::OutputArray &kx_, cv::OutputArray &ky_, const int& dx, const int& dy, const int& scale) {
  const int ksize = 3 + 2*(scale-1);

  if (scale == 1) {
    // The usual Scharr kernel
    cv::getDerivKernels(kx_,ky_,dx,dy,0,true,CV_32F);
    return;
  }

  kx_.create(ksize,1,CV_32F,-1,true);
  ky_.create(ksize,1,CV_32F,-1,true);
  cv::Mat kx = kx_.getMat();
  cv::Mat ky = ky_.getMat();

  CV_Assert( dx >= 0 && dy >= 0 && dx+dy == 1 );

  float w = 10.0/3.0;
  float norm = 1.0/(2.0*scale*(w+2.0));

  for (int k = 0; k < 2; k++) {
    cv::Mat* kernel = k == 0 ? &kx : &ky;
    int order = k == 0 ? dx : dy;
    float kerI[ksize];

    for (int t = 0; t<ksize; t++) {
      kerI[t] = 0;
    }

    if (order == 0) {
      kerI[0] = norm;
      kerI[ksize/2] = w*norm;
      kerI[ksize-1] = norm;
    }
    else if (order == 1) {
      kerI[0] = -1;
      kerI[ksize/2] = 0;
      kerI[ksize-1] = 1;
    }

    Mat temp(kernel->rows, kernel->cols, CV_32F, &kerI[0]);
    temp.copyTo(*kernel);
  }
}
