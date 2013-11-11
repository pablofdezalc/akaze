PROJECT(AKAZE_features)

cmake_minimum_required(VERSION 2.6)

if(COMMAND cmake_policy)
 cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Find external libraries
find_package(OpenCV REQUIRED)
find_package(OpenMP)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(CMAKE_BUILD_TYPE Release)
# Detect c++11 support
if(CMAKE_COMPILER_IS_GNUCXX)
   execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
   if (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
        message(STATUS "C++11 activated.")
        add_definitions("-std=gnu++11")
   elseif(GCC_VERSION VERSION_GREATER 4.3 OR GCC_VERSION VERSION_EQUAL 4.3)
        message(WARNING "C++0x activated. If you get any errors update to a compiler which fully supports C++11")
        add_definitions("-std=gnu++0x")
   else ()
        message(FATAL_ERROR "C++11 needed. Therefore a gcc compiler with a version higher than 4.3 is needed.")   
   endif()
else(CMAKE_COMPILER_IS_GNUCXX)
   add_definitions("-std=c++0x") 
endif(CMAKE_COMPILER_IS_GNUCXX)

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

INCLUDE_DIRECTORIES("src/lib/")
ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)

SET(SRCS
    src/lib/AKAZE.cpp
    src/lib/fed.cpp
    src/lib/utils.cpp
    src/lib/nldiffusion_functions.cpp
)

ADD_LIBRARY(AKAZE SHARED
  ${SRCS}
)

TARGET_LINK_LIBRARIES(AKAZE
  ${OpenCV_LIBS}
)

ADD_EXECUTABLE(
  akaze_features
  src/akaze_features.cpp
)

TARGET_LINK_LIBRARIES(
  akaze_features
  AKAZE
)

ADD_EXECUTABLE(
  akaze_match
  src/akaze_match.cpp
)

TARGET_LINK_LIBRARIES(
  akaze_match
  AKAZE
)

ADD_EXECUTABLE(
  akaze_compare
  src/akaze_compare.cpp
)

TARGET_LINK_LIBRARIES(
  akaze_compare
  AKAZE
)

install(TARGETS AKAZE DESTINATION /usr/local/akaze/lib)
install(FILES
    src/lib/AKAZE.h
    src/lib/fed.h
    src/lib/utils.h
    src/lib/nldiffusion_functions.h
    src/lib/config.h
    DESTINATION
    /usr/local/akaze/include)

################################

foreach(dir ${DIRS})
  FILE(GLOB files ${dir}/*.h*)
  LIST(APPEND h_files ${files})
endforeach(dir)
add_custom_target(TAGS etags --members -l c++ ${h_files} ${SRCS})
