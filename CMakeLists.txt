cmake_minimum_required(VERSION 3.12)
project(HOG_CV)

set(CMAKE_CXX_STANDARD 14)

add_executable(HOG_CV main.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( HOG_CV ${OpenCV_LIBS} )