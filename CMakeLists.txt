# ============================================================================ #
# [David OK] I want portable code so C++0x or C++11 features are not necessary.

cmake_minimum_required (VERSION 2.6)
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif ()

project (AKAZE_features)


# ============================================================================ #
# Where to output binaries and libraries
# 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# ============================================================================ #
# Find external libraries and dependencies.
find_package(OpenCV REQUIRED)
if (NOT APPLE)
  find_package(OpenMP)
endif ()
find_package(Doxygen)


# ============================================================================ #
# Set compiler flags.
#
# Unix specifics.
if (UNIX)
  add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)
endif ()
# Check For SSE features for the compiler.
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#include (CheckSSEFeatures)
# Activate OpenMP.
if (OPENMP_FOUND AND NOT APPLE)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
endif ()

# ============================================================================ #
# Create projects.
#
include_directories("src/lib/")
# 
# AKAZE libraries.
set(AKAZE_SRCS
    src/lib/config.h
    src/lib/AKAZE.h                  src/lib/AKAZE.cpp
    src/lib/fed.h                    src/lib/fed.cpp
    src/lib/nldiffusion_functions.h  src/lib/nldiffusion_functions.cpp)
add_library(AKAZE ${AKAZE_SRCS})
target_link_libraries(AKAZE ${OpenCV_LIBS})
# 
# OpenCV utilities.
set(OPENCV_UTILITIES_SRCS src/lib/utils.h src/lib/utils.cpp)
add_library(OpenCV_Utilities ${OPENCV_UTILITIES_SRCS})
target_link_libraries(OpenCV_Utilities ${OpenCV_LIBS})
# 
# Feature detector program.
add_executable(AKAZE_features
               src/akaze_features.h src/akaze_features.cpp)
target_link_libraries(AKAZE_features OpenCV_Utilities AKAZE)
#
# Image matching program.
add_executable(AKAZE_match
               src/akaze_match.h src/akaze_match.cpp)
target_link_libraries(AKAZE_match OpenCV_Utilities AKAZE)
# 
# Program that compares AKAZE with ORB and BRISK.
add_executable(AKAZE_compare
               src/akaze_compare.h src/akaze_compare.cpp)
target_link_libraries(AKAZE_compare OpenCV_Utilities AKAZE)
# 
# Generate Doxygen-based documentation project.
if(DOXYGEN_FOUND)
  add_custom_target(AKAZE_DOCUMENTATION
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)