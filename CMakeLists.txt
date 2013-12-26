project(AKAZE_features)

cmake_minimum_required (VERSION 2.6)
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif ()

# ============================================================================ #
# Where to output binaries and libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

# ============================================================================ #
# Where to install the AKAZE library
set(AKAZE_INSTALL_PREFIX "/usr/local/akaze/lib" CACHE PATH "Installation Directory")
set(AKAZE_INCLUDE_PREFIX "/usr/local/akaze/include" CACHE PATH "Includes Directory")

# ============================================================================ #
# Find external libraries and dependencies
find_package(OpenCV REQUIRED)
if (NOT APPLE)
  find_package(OpenMP)
endif ()
find_package(Doxygen)

# ============================================================================ #
# Compilation flags
IF(UNIX)
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
ENDIF(UNIX)

if(OPENMP_FOUND)
  MESSAGE("OpenMP found")
  if(UNIX)
    SET(CMAKE_C_FLAGS_RELEASE "-O3  -Wall -Wextra -Wunused-variable  -g -msse2 -msse3 -msse4 -ffast-math")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wunused-variable -g -msse2 -msse3 -msse4 -ffast-math")
  endif(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  else(OPENMP_FOUND)
    MESSAGE("OpenMP not found")
    if(UNIX)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wno-unknown-pragmas -Wunused-variable -g -msse2 -msse3 -msse4 -ffast-math")
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wno-unknown-pragmas -Wunused-variable -g -msse2 -msse3 -msse4 -ffast-math")
    endif(UNIX)
endif(OPENMP_FOUND)

# ============================================================================ #
# Create projects.
include_directories("src/lib/")

# AKAZE libraries
set(AKAZE_SRCS
    src/lib/config.h
    src/lib/AKAZE.h                  src/lib/AKAZE.cpp
    src/lib/fed.h                    src/lib/fed.cpp
    src/lib/nldiffusion_functions.h  src/lib/nldiffusion_functions.cpp
    src/lib/utils.h                  src/lib/utils.cpp)

add_library(AKAZE ${AKAZE_SRCS})
target_link_libraries(AKAZE ${OpenCV_LIBS})

# Feature detector program
add_executable(akaze_features
               src/akaze_features.h src/akaze_features.cpp)
target_link_libraries(akaze_features AKAZE)

# Image matching program
add_executable(akaze_match
               src/akaze_match.h src/akaze_match.cpp)
target_link_libraries(akaze_match AKAZE)

# Program that compares AKAZE with ORB and BRISK
add_executable(akaze_compare
               src/akaze_compare.h src/akaze_compare.cpp)
target_link_libraries(akaze_compare AKAZE)

# ============================================================================ #
# Library installation
install(TARGETS AKAZE DESTINATION ${AKAZE_INSTALL_PREFIX})
install(FILES
    src/lib/AKAZE.h
    src/lib/fed.h
    src/lib/utils.h
    src/lib/nldiffusion_functions.h
    src/lib/config.h
    DESTINATION
    ${AKAZE_INCLUDE_PREFIX}
)

# ============================================================================ #
# Generate Doxygen-based documentation project
if(DOXYGEN_FOUND)
  add_custom_target(akaze_documentation
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)
