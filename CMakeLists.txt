cmake_minimum_required(VERSION 3.16)
project(akaze)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
# Off by default so the library runs on any CPU of the target architecture (e.g. the conda package);
# turn it on for a local build tuned to this machine.
option(AKAZE_NATIVE "Optimise for the build machine's CPU (-march=native)" OFF)
if(AKAZE_NATIVE)
  add_compile_options(-march=native)
endif()
set (CMAKE_CXX_STANDARD 17)

add_compile_options(-Wextra -Wpedantic)

set(SRCS_AKAZE
  src/AKAZE.cpp
  src/fed.cpp
  src/nldiffusion_functions.cpp
  src/utils.cpp
)
  
find_package(OpenCV REQUIRED)
# OpenMP on every platform: Apple's own clang has no libomp, but conda's clang + llvm-openmp (what the conda
# package and AllFeature-VSLAM use on macOS) does, so the former `if(NOT APPLE)` guard only cost the parallel loops.
# Without OpenMP the sources still build single-threaded (the pragmas and omp_* calls sit behind _OPENMP).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "akaze: OpenMP ${OpenMP_CXX_VERSION}")
else()
  message(STATUS "akaze: OpenMP not found, single-threaded build")
endif()


add_library(${PROJECT_NAME} SHARED ${SRCS_AKAZE})
target_include_directories(${PROJECT_NAME} PUBLIC 
	"${PROJECT_SOURCE_DIR}"
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

configure_file(src/akaze.cmake.in
  "${PROJECT_BINARY_DIR}/akazeConfig.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION  ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION  ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION  ${CMAKE_INSTALL_PREFIX}/lib)
             
install(DIRECTORY
	"${PROJECT_SOURCE_DIR}/akaze"
        DESTINATION  ${CMAKE_INSTALL_PREFIX}/include
        FILES_MATCHING PATTERN "*.h")
                                                  
install(FILES "${PROJECT_BINARY_DIR}/akazeConfig.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/akaze/)  
