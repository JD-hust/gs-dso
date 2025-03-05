cmake_minimum_required(VERSION 3.21) 
set(PROJECT_NAME GS-DSO)

project(${PROJECT_NAME}) 
#set(CMAKE_VERBOSE_MAKEFILE ON)


#set(BUILD_TYPE Debug)
set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


set(GPU_RUNTIME "CUDA" CACHE STRING "HIP or CUDA or MPS")
set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE PATH "Path to the OPENCV installation directory")
set(OPENSPLAT_MAX_CUDA_COMPATIBILITY OFF CACHE BOOL "Build for maximum CUDA device compatibility")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(OPENSPLAT_BUILD_SIMPLE_TRAINER OFF CACHE BOOL "Build simple trainer applications")

# 服务器
# set(Torch_DIR "/home/dengjie/tools/libtorch/share/cmake/Torch")
# desktop-ubuntu
set(Torch_DIR  "/home/dj/tools/libtorch/libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118/libtorch/share/cmake/Torch")

file(READ "VERSION" APP_VERSION)

# Read git commit
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" 
                OUTPUT_VARIABLE GIT_REV 
                ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if (NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git commit ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()

add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# Don't complain about the override from NANOFLANN_BUILD_EXAMPLES
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# Use time-of-extraction for FetchContent'ed files modification time
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
# Suppress warning #20012-D (nvcc and glm)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -diag-suppress=20012 --expt-relaxed-constexpr)

# 获取外部依赖
include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip
)
FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
)
FetchContent_MakeAvailable(nlohmann_json nanoflann cxxopts)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    FetchContent_Declare(glm
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
    )
    FetchContent_MakeAvailable(glm)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)

endif()

# set gpu runtime
if(GPU_RUNTIME STREQUAL "CUDA")
    find_package(CUDAToolkit)
    if (NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA toolkit not found, building with CPU support only")
        set(GPU_RUNTIME "CPU")
    else() 
        if (OPENSPLAT_MAX_CUDA_COMPATIBILITY)
            execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-arch 
                    OUTPUT_VARIABLE LIST_GPU_ARCH 
                    ERROR_QUIET)
        endif()

        if(NOT LIST_GPU_ARCH AND OPENSPLAT_MAX_CUDA_COMPATIBILITY)
            message(WARNING "Cannot compile for max CUDA compatibility, nvcc does not support --list-gpu-arch")
            SET(OPENSPLAT_MAX_CUDA_COMPATIBILITY OFF)
        endif()
        if(NOT OPENSPLAT_MAX_CUDA_COMPATIBILITY)
            if(NOT CMAKE_CUDA_ARCHITECTURES)
                SET(CMAKE_CUDA_ARCHITECTURES 70;75;80)
            endif()
        else()
            # Build for maximum compatibility
            # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
            set(CMAKE_CUDA_ARCHITECTURES "")

            # Extract list of arch and gencodes
            string(REPLACE "\r" "" LIST_GPU_ARCH ${LIST_GPU_ARCH})
            string(REPLACE "\n" ";" LIST_GPU_ARCH ${LIST_GPU_ARCH})

            execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-code 
                OUTPUT_VARIABLE LIST_GPU_CODE 
                ERROR_QUIET)
            string(REPLACE "\r" "" LIST_GPU_CODE ${LIST_GPU_CODE})
            string(REPLACE "\n" ";" LIST_GPU_CODE ${LIST_GPU_CODE})

            list(GET LIST_GPU_CODE 0 TARGET_GPU_CODE)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${TARGET_GPU_CODE}")

            set(IDX 0)
            foreach(GPU_ARCH ${LIST_GPU_ARCH})
                string(REGEX MATCH "compute_([0-9]+)" GPU_ARCH_VERSION "${GPU_ARCH}")
                list(APPEND CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}")
                list(GET LIST_GPU_CODE ${IDX} GPU_CODE)
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=${GPU_ARCH},code=${GPU_CODE}")
                math(EXPR IDX "${IDX}+1")
            endforeach()
            message("Set CUDA flags: " ${CMAKE_CUDA_FLAGS})
        endif()
        # Set torch cuda architecture list
        set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
        list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "([0-9])([0-9])" "\\1.\\2")
        string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
        message(STATUS "** Updated TORCH_CUDA_ARCH_LIST to ${TORCH_CUDA_ARCH_LIST} **")
    endif()
elseif(GPU_RUNTIME STREQUAL "HIP")
    set(USE_HIP ON CACHE BOOL "Use HIP for GPU acceleration")

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)

    file(GLOB_RECURSE GSPLAT_GPU_SRC LIST_DIRECTORIES False rasterizer/gsplat/*.cu)
    set_source_files_properties(${GSPLAT_GPU_SRC} PROPERTIES LANGUAGE HIP)

    if(WIN32)
        set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
    else()
        set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
elseif(GPU_RUNTIME STREQUAL "MPS")
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
    message(STATUS "Metal framework found")

    set(XC_FLAGS -O3)
    set(USE_MPS ON CACHE BOOL "Use MPS for GPU acceleration")
else()
    set(GPU_RUNTIME "CPU")
endif()

set(CMAKE_CXX_STANDARD 17)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
    set(${GPU_RUNTIME}_STANDARD 17)
endif()

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)

add_subdirectory(GSModule)
add_subdirectory(DSO)
add_subdirectory(tests)