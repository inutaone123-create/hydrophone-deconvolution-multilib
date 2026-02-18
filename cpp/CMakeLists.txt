cmake_minimum_required(VERSION 3.14)
project(HydrophoneDeconvolution)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(deconvolution src/deconvolution.cpp)
target_include_directories(deconvolution PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(deconvolution PUBLIC Eigen3::Eigen)

# Test executable
add_executable(test_deconvolution tests/test_deconvolution.cpp)
target_link_libraries(test_deconvolution deconvolution)

# Export executable
add_executable(export_result tests/export_result.cpp)
target_link_libraries(export_result deconvolution)

# Pipeline library
add_library(pipeline src/pipeline.cpp)
target_include_directories(pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pipeline PUBLIC deconvolution Eigen3::Eigen)

# Pipeline export executable
add_executable(export_pipeline_result tests/export_pipeline_result.cpp)
target_link_libraries(export_pipeline_result pipeline)
