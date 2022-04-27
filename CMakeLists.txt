cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
set(CMAKE_CXX_STANDARD 14)

set(Torch_DIR /home/hove/Backup/software_backup/cuda10.2/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_DIRS})

# add_executable(example-app example-app.cpp)
# target_link_libraries(example-app ${TORCH_LIBRARIES} ${OpenCV_LIBS})


include_directories(
        ${PROJECT_SOURCE_DIR}/include)
# FILE(GLOB ALL_SOURCES "./src/FastScnn_v3.cc" "./include/FastScnn_v3.h")
add_executable(inference inference.cpp src/FastScnn_v3.cc)
target_link_libraries(inference ${TORCH_LIBRARIES} ${OpenCV_LIBS})