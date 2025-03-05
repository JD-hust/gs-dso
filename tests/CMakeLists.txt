# simple_trainer
if(OPENSPLAT_BUILD_SIMPLE_TRAINER)
    message("building simple trainer")
    add_executable(simple_trainer simple_trainer.cpp 
        ${PROJECT_SOURCE_DIR}/GSModule/src/project_gaussians.cpp 
        ${PROJECT_SOURCE_DIR}/GSModule/src/rasterize_gaussians.cpp 
        ${PROJECT_SOURCE_DIR}/GSModule/src/cv_utils.cpp)
    install(TARGETS simple_trainer DESTINATION bin)
    target_include_directories(simple_trainer PRIVATE
        ${PROJECT_SOURCE_DIR}/GSModule/rasterizer
        ${GPU_INCLUDE_DIRS}
    )
    target_link_libraries(simple_trainer ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
    target_link_libraries(simple_trainer 
        nlohmann_json::nlohmann_json
        cxxopts::cxxopts
        nanoflann::nanoflann
    )
    if (NOT WIN32)
        target_link_libraries(simple_trainer pthread)
    endif()
    set_property(TARGET simple_trainer PROPERTY CXX_STANDARD 17)
    if(GPU_RUNTIME STREQUAL "HIP")
        target_compile_definitions(simple_trainer PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    elseif(GPU_RUNTIME STREQUAL "CUDA")
        target_compile_definitions(simple_trainer PRIVATE USE_CUDA)
    elseif(GPU_RUNTIME STREQUAL "MPS")
        target_compile_definitions(simple_trainer PRIVATE USE_MPS)
    endif()
endif()

# 测试多线程机制
add_executable(gs_for_dso gs_for_dso.cpp)
target_link_libraries(gs_for_dso PUBLIC ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS} gsmodule) 
target_link_libraries(gs_for_dso PRIVATE
   nlohmann_json::nlohmann_json
   cxxopts::cxxopts
   nanoflann::nanoflann
)
target_include_directories(gs_for_dso PRIVATE
    ${PROJECT_SOURCE_DIR}/GSModule/rasterizer
    ${GPU_INCLUDE_DIRS}
)