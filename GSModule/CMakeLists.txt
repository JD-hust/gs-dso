# 跨平台支持
if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()
set(Torch_DIR "/home/dj/tools/libtorch/libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)

# 在非 Windows 和非 macOS 系统（如 Linux）上，显式地指定 CUDA 编译器为 nvcc；跨平台构建项目
if (NOT WIN32 AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()


# 设置变量 GSPLAT_LIBS，存储高斯泼溅的实现库
set(GSPLAT_LIBS gsplat_cpu)

# gpu runtime print
message(STATUS "GPU runtime: ${GPU_RUNTIME}")

# gsplat
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    # gsplat 是 cuda加速 库
    add_library(gsplat rasterizer/gsplat/forward.cu rasterizer/gsplat/backward.cu rasterizer/gsplat/bindings.cu rasterizer/gsplat/helpers.cuh)
    list(APPEND GSPLAT_LIBS gsplat)
    if(GPU_RUNTIME STREQUAL "CUDA")
        set(GPU_LIBRARIES "cuda")
        target_link_libraries(gsplat PUBLIC cuda) 
    else(GPU_RUNTIME STREQUAL "HIP")
        set(GPU_INCLUDE_DIRS "${ROCM_ROOT}/include")
        target_compile_definitions(gsplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    endif()
    target_include_directories(gsplat PUBLIC 
        rasterizer
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(gsplat PUBLIC glm::glm-header-only) 
    set_target_properties(gsplat PROPERTIES LINKER_LANGUAGE CXX) # 设置目标库的属性
elseif(GPU_RUNTIME STREQUAL "MPS")    
    add_library(gsplat rasterizer/gsplat-metal/gsplat_metal.mm)
    list(APPEND GSPLAT_LIBS gsplat)
    target_link_libraries(gsplat PRIVATE 
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
    target_include_directories(gsplat PRIVATE ${TORCH_INCLUDE_DIRS})
    # copy shader files to bin directory
    configure_file(rasterizer/gsplat-metal/gsplat_metal.metal ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.metal COPYONLY)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
        COMMAND xcrun -sdk macosx metal    ${XC_FLAGS} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.metal -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.air
        COMMAND xcrun -sdk macosx metallib                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.air   -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
        COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.air
        COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.metal
        DEPENDS rasterizer/gsplat-metal/gsplat_metal.metal
        COMMENT "Compiling Metal kernels"
    )

    add_custom_target(
        gsplat_metal ALL
        DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    )
endif()
# gsplat_cpu
add_library(gsplat_cpu rasterizer/gsplat-cpu/gsplat_cpu.cpp) 
target_include_directories(gsplat_cpu PRIVATE ${TORCH_INCLUDE_DIRS})


# gsmodule
file(GLOB_RECURSE source CONFIGURE_DEPENDS src/*.cpp)
set(GSModule_LIBS "")
add_library(gsmodule STATIC ${source}) 
add_library(GSModule::gsmodule ALIAS gsmodule)
list(APPEND GSModule_LIBS gsmodule)
target_include_directories(gsmodule PUBLIC
    rasterizer
    include/GSModule
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(gsmodule 
    ${STDPPFS_LIBRARY}
    ${GPU_LIBRARIES}
    ${GSPLAT_LIBS}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_link_libraries(gsmodule 
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)

if(GPU_RUNTIME STREQUAL "HIP")
    target_compile_definitions(gsmodule PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
elseif(GPU_RUNTIME STREQUAL "CUDA")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
    target_compile_definitions(gsmodule PRIVATE USE_CUDA)
elseif(GPU_RUNTIME STREQUAL "MPS")
    target_compile_definitions(gsmodule PRIVATE USE_MPS)
endif()

# opensplat
add_executable(opensplat opensplat.cpp) 
install(TARGETS opensplat DESTINATION bin) 
target_include_directories(opensplat PUBLIC
    rasterizer
    include/GSModule
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(opensplat 
    ${GSModule_LIBS}
    ${STDPPFS_LIBRARY}
    ${GPU_LIBRARIES}
    ${GSPLAT_LIBS}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_link_libraries(opensplat 
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)

if (NOT WIN32)
    target_link_libraries(opensplat pthread)
endif()


if(GPU_RUNTIME STREQUAL "HIP")
    target_compile_definitions(opensplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
elseif(GPU_RUNTIME STREQUAL "CUDA")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
    target_compile_definitions(opensplat PRIVATE USE_CUDA)
elseif(GPU_RUNTIME STREQUAL "MPS")
    target_compile_definitions(opensplat PRIVATE USE_MPS)
endif()


