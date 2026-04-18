cmake_minimum_required(VERSION 3.22.1)
project(cunyi_doctor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 关闭 llama.cpp 不需要的组件
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# 只启用 CPU 后端
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)

# llama.cpp (local source via submodule)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-src llama_build)

add_library(cunyi_doctor SHARED
    llama_jni.cpp
)

target_include_directories(cunyi_doctor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-src
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp-src/ggml/include
)

target_link_libraries(cunyi_doctor
    PRIVATE
    llama
    ggml
    log
)
