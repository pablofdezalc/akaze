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
 */

#include "akaze_match.h"

// Namespaces
using namespace std;
using namespace cv;

// Some image matching options
const bool COMPUTE_HOMOGRAPHY = false;	// 0->Use ground truth homography, 1->Estimate homography with RANSAC
const float MIN_H_ERROR = 2.50f;	// Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;		// NNDR Matching value

//*************************************************************************************
//*************************************************************************************

/** Main Function 																	 */
int main( int argc, char *argv[] ) {

    // Variables
    AKAZEOptions options;
    Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
    char img_name1[NMAX_CHAR], img_name2[NMAX_CHAR], hfile[NMAX_CHAR];
    char rfile[NMAX_CHAR];
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
    if (parse_input_options(options,img_name1,img_name2,hfile,rfile,argc,argv)) {
        return -1;
    }

    // Read the image, force to be grey scale
    img1 = imread(img_name1,0);

    if (img1.data == NULL) {
        cout << "Error loading image: " << img_name1 << endl;
        return -1;
    }

    // Read the image, force to be grey scale
    img2 = imread(img_name2,0);

    if (img2.data == NULL) {
        cout << "Error loading image: " << img_name2 << endl;
        return -1;
    }

    // Convert the images to float
    img1.convertTo(img1_32,CV_32F,1.0/255.0,0);
    img2.convertTo(img2_32,CV_32F,1.0/255.0,0);

    // Color images for results visualization
    img1_rgb = Mat(Size(img1.cols,img1.rows),CV_8UC3);
    img2_rgb = Mat(Size(img2.cols,img1.rows),CV_8UC3);
    img_com = Mat(Size(img1.cols*2,img1.rows),CV_8UC3);
    img_r = Mat(Size(img_com.cols*rfactor,img_com.rows*rfactor),CV_8UC3);

    // Read the homography file
    read_homography(hfile,HG);

    // Create the first HKAZE object
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

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * and image matching between two input images
 * @param options Structure that contains KAZE settings
 * @param img_name1 Name of the first input image
 * @param img_name2 Name of the second input image
 * @param hom Name of the file that contains a ground truth homography
 * @param kfile Name of the file where the keypoints where be stored
 */
int parse_input_options(AKAZEOptions &options, char *img_name1, char *img_name2, char *hom,
                        char *kfile, int argc, char *argv[] ) {

    // If there is only one argument return
    if (argc == 1) {
        show_input_options_help(1);
        return -1;
    }
    // Set the options from the command line
    else if (argc >= 2) {

        // Load the default options
        options = AKAZEOptions();
        strcpy(img_name1,argv[1]);
        strcpy(img_name2,argv[2]);
        strcpy(hom,argv[3]);
        strcpy(kfile,"./results.txt");

        for (int i = 1; i < argc; i++) {
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
                if (i >= argc) {
                    cout << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.omax = atof(argv[i]);
                }
            }
            else if ( !strcmp(argv[i],"--dthreshold")) {
                i = i+1;
                if (i >= argc) {
                    cout << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.dthreshold = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i],"--dthreshold2")) {
                i = i+1;
                if (i >= argc) {
                    cout << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.dthreshold2 = atof(argv[i]);
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
            else if (!strcmp(argv[i],"--diffusivity"))
            {
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

                    if ( options.descriptor_channels <= 0 || options.descriptor_channels > 3 ) {
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

                    if ( options.descriptor_size < 0 ) {
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
            else if (!strcmp(argv[i],"--kfile")) {
                i = i+1;
                if (i >= argc) {
                    cout << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    strcpy(kfile,argv[i]);
                }
            }
            else if (!strcmp(argv[i],"--verbose")) {
                options.verbosity = true;
            }
            else if (!strcmp(argv[i],"--help")) {
                // Show the help!!
                show_input_options_help(1);
                return -1;
            }
            else if (!strncmp(argv[i],"--",2))
                cout << "Unknown command "<<argv[i] << endl;
        }
    }
    else {
        cout << "Error introducing input options!!" << endl;

        // Show the help!!
        show_input_options_help(1);
        return -1;
    }

    return 0;
}
