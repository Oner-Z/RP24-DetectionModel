cmake_minimum_required(VERSION 3.14)
project(RP24-DetectionModel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 OpenVINO
find_package(OpenVINO REQUIRED)

# 包含头文件路径（如有自定义头文件目录可添加）
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenVINO_INCLUDE_DIRS})

# 源文件
set(SOURCES
    main.cpp
    OpenvinoInfer.cpp
)

add_executable(rp24_detect ${SOURCES})

target_link_libraries(rp24_detect
    ${OpenCV_LIBS}
    openvino::runtime
)