cmake_minimum_required (VERSION 3.10.0)
project(
    "TinyTorch"
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

find_package(CUDAToolkit 11.4 REQUIRED)

# Make all executables goto bin directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    message("Building in DEBUG mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -O1 -fsanitize=address -Wno-unused-parameter -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG -arch=native -fPIC")
elseif (CMAKE_BUILD_TYPE MATCHES PROFILE)
    message("Building in PROFILE mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -DNDEBUG -pg -Wno-unused-parameter -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --use_fast_math -arch=native -fPIC")
else()
    message("Building in RELEASE mode")
    # NOTE. "assert" won't be evaluated when NDEBUG is defined
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -DNDEBUG -Wno-unused-parameter -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --use_fast_math -arch=native -fPIC")
endif()

find_package(pybind11 REQUIRED)

# # Set up COMMON_HEADER_DIRS and COMMON_LIB_DIRS
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
)
set(COMMON_LIB_DIRS "")
set(COMMON_LIBS CUDA::cudart)

# Let COMMON_HEADER_DIRS & COMMON_LIB_DIRS take effect
include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})
link_libraries(${COMMON_LIBS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)