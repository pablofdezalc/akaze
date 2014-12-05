
# Find external libraries and dependencies
find_package(OpenCV REQUIRED)
find_package(Doxygen)

if (NOT APPLE)
  find_package(OpenMP)
endif ()

# ============================================================================ #
# Compilation flags
IF(UNIX)
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
ENDIF(UNIX)

if(OPENMP_FOUND)
  MESSAGE("OpenMP found")
  if(UNIX)
    SET(CMAKE_C_FLAGS_RELEASE "-O3  -Wall -Wextra -Wunused-variable  -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
  endif(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  else(OPENMP_FOUND)
    MESSAGE("OpenMP not found")
    if(UNIX)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wall -std=c++0x -Wunused-variable -Wno-unknown-pragmas -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++0x -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    endif(UNIX)
endif(OPENMP_FOUND)

# ============================================================================ #
# Create projects
include_directories("lib/")

# AKAZE libraries
set(AKAZE_SRCS
    lib/AKAZEConfig.h
    lib/AKAZE.h                  lib/AKAZE.cpp
    lib/fed.h                    lib/fed.cpp
    lib/nldiffusion_functions.h  lib/nldiffusion_functions.cpp
    lib/utils.h                  lib/utils.cpp)

add_library(AKAZE ${AKAZE_SRCS})
target_link_libraries(AKAZE ${OpenCV_LIBS})

# Feature detector program
add_executable(akaze_features akaze_features.cpp)
target_link_libraries(akaze_features AKAZE)

# Image matching program
add_executable(akaze_match akaze_match.cpp)
target_link_libraries(akaze_match AKAZE)

# Program that compares AKAZE with ORB and BRISK
add_executable(akaze_compare akaze_compare.cpp)
target_link_libraries(akaze_compare AKAZE)

# ============================================================================ #
# Library installation
install(TARGETS AKAZE DESTINATION ${AKAZE_INSTALL_PREFIX})
install(FILES
    lib/AKAZE.h
    lib/fed.h
    lib/utils.h
    lib/nldiffusion_functions.h
    lib/AKAZEConfig.h
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
