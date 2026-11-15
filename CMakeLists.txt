cmake_minimum_required(VERSION 4.4.3)

project(ITLab26)

add_executable(numerical methods)
#differential equations
#computational tasks

target_sources(numerical methods PRIVATE Euler.cxx)


