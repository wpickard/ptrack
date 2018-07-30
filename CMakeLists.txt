#  minimum required cmake version: 3.8.0
cmake_minimum_required(VERSION 3.8.0)

project(ptrack)

# Required packages
find_package(OpenCV)
find_package(realsense2)
find_package(RapaPololuMaestro)

add_executable(ptrack ptrack.cpp cv-helpers.hpp)

# add include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${realsense_INCLUDE_DIR})

target_link_libraries(ptrack ${OpenCV_LIBS} ${realsense2_LIBRARY} /usr/local/lib/libRapaPololuMaestro.a)

set(PROTOTEXT_FILE "MobileNetSSD_deploy.prototxt")
set(PROTOTEXT_URL "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt")
set(PROTOTEXT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PROTOTEXT_FILE}")

set(MODEL_FILE "MobileNetSSD_deploy.caffemodel")
set(MODEL_URL "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc")
set(MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_FILE}")

if(NOT EXISTS "${PROTOTEXT_PATH}")
	message("Downloading ${PROTOTEXT_FILE} into ${CMAKE_CURRENT_BINARY_DIR}")
    file(DOWNLOAD "${PROTOTEXT_URL}" "${PROTOTEXT_PATH}")
endif()

if(NOT EXISTS "${MODEL_PATH}")
	message("Downloading ${MODEL_FILE} into ${CMAKE_CURRENT_BINARY_DIR}")
    file(DOWNLOAD "${MODEL_URL}" "${MODEL_PATH}")
endif()
