cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(resnet_run)

set(Torch_DIR "./libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

add_executable(resnet_run main.cpp)

target_link_libraries(resnet_run "${TORCH_LIBRARIES}")

set_property(TARGET resnet_run PROPERTY CXX_STANDARD 17)
