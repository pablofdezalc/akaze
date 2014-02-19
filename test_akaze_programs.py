import sys
import os
import subprocess

# ===============================================================================
# Populate AKAZE-based programs
programs = ['akaze_features', 'akaze_match', 'akaze_compare']
for i in range(len(programs)):
  programs[i] = os.path.join('.', 'bin' , 'Release' , programs[i])

# ===============================================================================
# Helper functions
def extract_AKAZE_features(imagePath):
  command = " ".join([programs[0], ' ', imagePath])
  os.system(command)

def match_AKAZE_features(imagePath1, imagePath2, gdTruthHomography):
  command = " ".join(
    [programs[1], ' ', imagePath1, ' ', imagePath2, ' ', gdTruthHomography] )
  os.system(command)

def compare_AKAZE_BRISK_ORB(imagePath1, imagePath2, gdTruthHomography):
  command = " ".join(
    [programs[2], ' ', imagePath1, ' ', imagePath2, ' ', gdTruthHomography] )
  os.system(command)

# ===============================================================================
# Example datasets
imagePath1 = os.path.join('.','datasets','iguazu', 'img1.pgm')
imagePath2 = os.path.join('.','datasets','iguazu', 'img4.pgm')
gdTruthHomography = os.path.join('.','datasets','iguazu', 'H1to4p')

# Go!
extract_AKAZE_features(imagePath1)
match_AKAZE_features(imagePath1, imagePath2, gdTruthHomography)
compare_AKAZE_BRISK_ORB(imagePath1, imagePath2, gdTruthHomography)
