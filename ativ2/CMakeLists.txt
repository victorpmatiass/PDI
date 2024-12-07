cmake_minimum_required(VERSION 3.5)
project(ativ2)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Executável para Laplacian
add_executable(main_laplacian src/main_laplacian.cpp src/laplacian.cpp src/calcHist.cpp)
target_link_libraries(main_laplacian ${OpenCV_LIBS})

# Executável para Mean Filter
add_executable(main_mean_filter src/main_mean_filter.cpp src/mean_filter.cpp)
target_link_libraries(main_mean_filter ${OpenCV_LIBS})
