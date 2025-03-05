set(Torch_DIR  "/home/dj/tools/libtorch/libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 

# optional libraries
find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)
find_package(OpenCV QUIET)

add_definitions("-DENABLE_SSE")

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

set(dso_SOURCE_FILES
  src/FullSystem/FullSystem.cpp
  src/FullSystem/FullSystemOptimize.cpp
  src/FullSystem/FullSystemOptPoint.cpp
  src/FullSystem/FullSystemDebugStuff.cpp
  src/FullSystem/FullSystemMarginalize.cpp
  src/FullSystem/Residuals.cpp
  src/FullSystem/CoarseTracker.cpp
  src/FullSystem/CoarseInitializer.cpp
  src/FullSystem/ImmaturePoint.cpp
  src/FullSystem/HessianBlocks.cpp
  src/FullSystem/PixelSelector2.cpp
  src/OptimizationBackend/EnergyFunctional.cpp
  src/OptimizationBackend/AccumulatedTopHessian.cpp
  src/OptimizationBackend/AccumulatedSCHessian.cpp
  src/OptimizationBackend/EnergyFunctionalStructs.cpp
  src/util/settings.cpp
  src/util/Undistort.cpp
  src/util/globalCalib.cpp
)

# decide if we have pangolin
if (Pangolin_FOUND)
	message("--- found PANGOLIN, compiling dso_pangolin library.")
	include_directories( ${Pangolin_INCLUDE_DIRS} ) 
	set(dso_pangolin_SOURCE_FILES 
	  src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	  src/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
	message("    this means there will be no 3D display / GUI available for dso_dataset.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling dso_opencv library.")
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	set(dso_opencv_SOURCE_FILES 
	  src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	  src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(dso_opencv_SOURCE_FILES 
	  src/IOWrapper/ImageDisplay_dummy.cpp
	  src/IOWrapper/ImageRW_dummy.cpp)
	set(HAS_OPENCV 0)
endif ()

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

# gs_dso_dataset
file(GLOB_RECURSE GSDSO_SOURCES CONFIGURE_DEPENDS
    src/main_dso_pangolin_with_gs.cpp
)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling gs_dso_dataset.")
	add_executable(gsdso_dataset ${dso_SOURCE_FILES} ${GSDSO_SOURCES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})
    
    target_include_directories(gsdso_dataset PUBLIC
    include
    ${PROJECT_SOURCE_DIR}/GSModule/include/GSModule
    ${PROJECT_SOURCE_DIR}/GSModule/rasterizer
    ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
    ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
    ${GPU_INCLUDE_DIRS} 
    ${EIGEN3_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ) 

    target_link_libraries(gsdso_dataset PUBLIC boost_system cxsparse ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} gsmodule)
    target_link_libraries(gsdso_dataset PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann)

    if (NOT WIN32)
        target_link_libraries(gsdso_dataset PUBLIC pthread)
    endif()

    if(GPU_RUNTIME STREQUAL "HIP")
        target_compile_definitions(gsdso_dataset PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    elseif(GPU_RUNTIME STREQUAL "CUDA")
        target_compile_definitions(gsdso_dataset PRIVATE USE_CUDA)
    elseif(GPU_RUNTIME STREQUAL "MPS")
        target_compile_definitions(gsdso_dataset PRIVATE USE_MPS)
    endif()
elseif()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()

