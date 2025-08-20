cmake_minimum_required(VERSION 3.14)

project(loglikelihoodpython CXX)

# Find Python and Pybind11
# This will find the Python installation and the pybind11 package.
# It automatically sets up include directories and link libraries.
find_package(pybind11 REQUIRED)

# Add your pybind11 module
# `example` is the name of the module that will be imported in Python.
# `example.cpp` is your source file.
pybind11_add_module(hiv_smm loglikelihoodpython.cpp)

# Optional: Set C++ standard
# pybind11 requires at least C++14, but C++17 is a good practice.
target_compile_features(hiv_smm PRIVATE cxx_std_17)