cmake_minimum_required(VERSION 3.20)
project(RoughSDE)

# Use C++20 as requested
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# We need pybind11
find_package(pybind11 REQUIRED)

# The user's MacTensor library location
set(MACTENSOR_DIR "../MacTensor")
include_directories(${MACTENSOR_DIR}/include)
include_directories(${MACTENSOR_DIR}/src)

# Define the python extension module
pybind11_add_module(rough_sde 
    src/data_gen.cpp 
    src/signatures.cpp 
    src/sde_solver.cpp 
    src/bindings.cpp
    ${MACTENSOR_DIR}/src/core/matrix.cpp
)

# MacTensor uses Accelerate framework and standard C++
target_link_libraries(rough_sde PRIVATE "-framework Accelerate")
