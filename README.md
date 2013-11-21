A-KAZE Features
==========

## Code Documentation

See http://davidok8.github.io/AKAZE/index.html

## Introduction

From the web page: http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html

"**KAZE Features** is a novel 2D feature detection and description method that operates completely in a nonlinear scale space. Previous methods such as SIFT or SURF find features in the Gaussian scale space (particular instance of linear diffusion). However, Gaussian blurring does not respect the natural boundaries of objects and smoothes in the same degree details and noise when evolving the original image through the scale space.

By means of nonlinear diffusion we can detect and describe features in nonlinear scale spaces keeping important image details and removing noise as long as we evolve the image in the scale space. We use variable conductance diffusion which is one of the simplest cases of nonlinear diffusion. The nonlinear scale space is build efficiently by means of Additive Operator Splitting (AOS) schemes, which are stable for any step size and are parallelizable.

**Accelerated-KAZE Features** uses a novel mathematical framework called **Fast Explicit Diffusion (FED)** embedded in a pyramidal framework to speed-up dramatically the nonlinear scale space computation. In addition, we compute a robust **Modified-Local Difference Binary (M-LDB)** descriptor that exploits gradient information from the nonlinear scale space. A-KAZE obtains comparable results to KAZE in some datasets, while being several orders of magnitude faster.

Our results reveal a big improvement in repeatability and distinctiviness, for common 2D image matching applications.

**Important**: If you work in a research institution, university, company or you are a freelance and you are using KAZE or A-KAZE in your work, please let [Pablo F. Alcantarilla] know and send [him] an email...
[Pablo F. Alcantarilla] would like to know the people that are using KAZE around the world!!"

email: pablofdezalc@gmail.com


## CHANGELOG

**Version: 1.0.0_DO_1** [Source code slightly modified by _David OK (david.ok8@gmail.com)_]
- Date: 18-10-2013
- Changes:
  * Code cleaning, moved DOXYGEN comments to header files.
  * Minor fix
  * Created separate library for utils[.h, .cpp]
- TODO:
  * Re-add `-ffast-math` for UNIX-based system?
  * Remove warnings.
  * Generate shared library instead?

**Version: 1.0.0_DO** [Source code slightly modified by _David OK (david.ok8@gmail.com)_]
- Date: 17-10-2013
- Changes:
  * Modified slightly source code to compile it with **MSVC 11 x64** and **OpenMP**.
  * Modified `CMakeLists.txt` to generate static library instead of shared library.
  * Removed C support in `CMakeLists.txt` because the source code uses the C++ STL.
  * Added CMake module in order to enable **SSE features** in a **cross-platform** fashion.
  * Added Doxygen documentation generation in `CMakeLists.txt`
- TODO:
  * Test the A-KAZE works correctly.
  * Re-add `-ffast-math` for UNIX-based system?
  * Remove warnings.
  * Generate shared library instead?


**Version: 1.0.0**
- Date: 16-09-2013
- Changes:
  * Initial Release


## Library Dependencies

The code is mainly based on the **OpenCV** library using the C++ interface.

In order to compile the code, the following libraries to be installed on your system:
- **OpenCV version 2.4.0 or higher**
- **CMake version 2.6 or higher**

If you want to use **OpenMP** parallelization you will need to install OpenMP in your system
In Linux you can do this by installing the **gomp** library

You will also need **doxygen** in case you need to generate the documentation

Tested compilers:
- GCC 4.2-4.7
- MSVC 11 x64

Tested systems:
- Ubuntu 11.10, 12.04, 12.10
- Kubuntu 10.04
- Windows 8

## Getting Started

Compiling:

1. `$ mkdir build`
2. `$ cd build>`
3. `$ cmake ..`
4. `$ make`

Additionally you can also install the library in `/usr/local/akaze/lib` by typing:
`$ sudo make install`

If the compilation is successful you should see three executables in the folder bin:
- `akaze_features`
- `akaze_match`
- `akaze_compare`

Additionally, the library `libAKAZE[.a, .lib]` will be created in the folder `lib`.

If there is any error in the compilation, perhaps some libraries are missing.
Please check the Library dependencies section.

Examples:
To see how the code works, examine the three examples provided.

## Documentation
In the working folder, type: `doxygen`

The documentation will be generated in the ./doc folder

## Computing A-KAZE Features

For running the program you need to type in the command line the following arguments:
`./akaze_features img.jpg [options]`

The `[options]` are not mandatory. In case you do not specify additional options, default arguments will be
used. Here is a description of the additional options:

- `--verbose`: if verbosity is required
- `--help`: for showing the command line options
- `--soffset`: the base scale offset (sigma units)
- `--omax`: the coarsest nonlinear scale space level (sigma units)
- `--nsublevels`: number of sublevels per octave
- `--diffusivity`: diffusivity function `0` -> Perona-Malik 1, `1` -> Perona-Malik 2, `2` -> Weickert
- `--dthreshold`: Feature detector threshold response for accepting points
- `--descriptor`: Descriptor Type `0` -> SURF, `1` -> M-SURF, `2` -> M-LDB
- `--upright`: `0` -> Rotation Invariant, `1` -> No Rotation Invariant
- `--descriptor_channels`: Descriptor Channels for M-LDB. Valid values: 1, 2 (intensity+gradient magnitude), 3(intensity + X and Y gradients)
- `--descriptor_size`: Descriptor size for M-LDB in bits. 0 means the full length descriptor (486). Any other value will use a random bit selection
- `--show_results`: `1` in case we want to show detection results. `0` otherwise

## Important Things:

* Check `config.h` in case you would like to change the value of some default settings
* The **k** constrast factor is computed as the 70% percentile of the gradient histogram of a
smoothed version of the original image. Normally, this empirical value gives good results, but
depending on the input image the diffusion will not be good enough. Therefore I highly
recommend you to visualize the output images from save_scale_space and test with other k
factors if the results are not satisfactory

## Image Matching Example with A-KAZE Features

The code contains one program to perform image matching between two images.
If the ground truth transformation is not provided, the program estimates a fundamental matrix or a planar homography using
RANSAC between the set of correspondences between the two images.

For running the program you need to type in the command line the following arguments:
`./akaze_match img1.jpg img2.pgm homography.txt [options]`

The `[options]` are not mandatory. In case you do not specify additional options, default arguments will be
used. 

The datasets folder contains the **Iguazu** dataset described in the paper and additional datasets from Mikolajczyk et al. evaluation.
The **Iguazu** dataset was generated by adding Gaussian noise of increasing standard deviation.

For example, with the default configuration parameters used in the current code version you should get
the following results:

```
./akaze_match ../../datasets/iguazu/img1.pgm 
              ../../datasets/iguazu/img4.pgm 
              ../../datasets/iguazu/H1to4p
              --descriptor 2
```

```
Number of Keypoints Image 1: 1137
Number of Keypoints Image 2: 1046
KAZE Features Extraction Time (ms): 228.145
Matching Descriptors Time (ms): 41.3758
Homography Computation Time (ms): 0.028648
Number of Matches: 665
Number of Inliers: 605
Number of Outliers: 60
Inliers Ratio: 90.9774
```

## Image Matching Comparison between A-KAZE, ORB and BRISK (OpenCV)

The code contains one program to perform image matching between two images, showing a comparison between A-KAZE features, ORB
and BRISK. All these implementations are based on the OpenCV library. 

The program assumes that the ground truth transformation is provided

For running the program you need to type in the command line the following arguments:
```
./akaze_compare img1.jpg img2.pgm homography.txt [options]
```

For example, running kaze_compare with the first and third images from the boat dataset you should get the following results:

```
./akaze_compare ../../datasets/boat/img1.pgm 
                ../../datasets/boat/img3.pgm
                ../../datasets/boat/H1to3p
                --dthreshold 0.004
                --dthreshold2 0.004
                --diffusivity 1
                --descriptor 2
                --nsublevels 3
```

```
ORB Results
**************************************
Number of Keypoints Image 1: 1510
Number of Keypoints Image 2: 1516
Number of Matches: 304
Number of Inliers: 277
Number of Outliers: 27
Inliers Ratio: 91.1184
ORB Features Extraction Time (ms): 74.603

BRISK Results
**************************************
Number of Keypoints Image 1: 3457
Number of Keypoints Image 2: 3031
Number of Matches: 159
Number of Inliers: 116
Number of Outliers: 43
Inliers Ratio: 72.956
BRISK Features Extraction Time (ms): 482.781

A-KAZE Results
**************************************
Number of Keypoints Image 1: 1549
Number of Keypoints Image 2: 1193
Number of Matches: 414
Number of Inliers: 351
Number of Outliers: 63
Inliers Ratio: 84.7826
A-KAZE Features Extraction Time (ms): 230.502
```

**A-KAZE features** is **open source** and you can use that **freely even in commercial applications**. While A-KAZE is a bit slower compared to **ORB** and **BRISK**, it provides much better performance. In addition, for images with small resolution such as 640x480 the algorithm can run in real-time. In the next future we plan to release a GPGPU implementation.

## Citation

If you use this code as part of your work, please cite the following papers:

1. **Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces**. Pablo F. Alcantarilla, J. Nuevo and Adrien Bartoli. _In British Machine Vision Conference (BMVC), Bristol, UK, September 2013_

2. **KAZE Features**. Pablo F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. _In European Conference on Computer Vision (ECCV), Fiorenze, Italy, October 2012_

## Contact Info

In case you have any question, find any bug in the code or want to share some improvements, please contact:
Pablo F. Alcantarilla
email: pablofdezalc@gmail.com

