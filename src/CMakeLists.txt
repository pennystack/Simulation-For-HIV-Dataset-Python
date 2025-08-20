cmake_minimum_required(VERSION 3.14)

project(loglikelihoodpython CXX)
find_package(pybind11 REQUIRED)

pybind11_add_module(hiv_smm loglikelihoodpython.cpp)

target_compile_features(hiv_smm PRIVATE cxx_std_17)
