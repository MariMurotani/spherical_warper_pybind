cmake_minimum_required(VERSION 3.4...3.18)
project(spherical_warper)

# 強制的にPython 3.13を指定（なくてもOKだがあると安全）
#set(Python3_EXECUTABLE "/Users/ebiharamari/.pyenv/versions/3.13.0/bin/python")
set(Python3_EXECUTABLE "/Users/marimurotani/.pyenv/versions/3.13.0/envs/spherical-warp-env/bin/python")


find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(spherical_warper spherical_warper.cpp)
target_link_libraries(spherical_warper PRIVATE ${OpenCV_LIBS})