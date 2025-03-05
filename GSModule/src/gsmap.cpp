#include <filesystem>
#include "gsmap.hpp"
#include "model.hpp"
#include "constants.hpp"
#include "tile_bounds.hpp"
#include "project_gaussians.hpp"
#include "rasterize_gaussians.hpp"
#include "tensor_math.hpp"
#include "gsplat.hpp"
#include "cv_utils.hpp"

#ifdef USE_HIP
#include <c10/hip/HIPCachingAllocator.h>
#elif defined(USE_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace fs = std::filesystem;

torch::Tensor GSMap::forward(Camera& cam, int step, bool depth_flag){
    const float scaleFactor = getDownscaleFactor(step);
    const float fx = cam.fx / scaleFactor;
    const float fy = cam.fy / scaleFactor;
    const float cx = cam.cx / scaleFactor;
    const float cy = cam.cy / scaleFactor;
    const int height = static_cast<int>(static_cast<float>(cam.height) / scaleFactor);
    const int width = static_cast<int>(static_cast<float>(cam.width) / scaleFactor);

    torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)});

    R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));

    // worldToCam
    torch::Tensor Rinv = R.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T);
    
    lastHeight = height;
    lastWidth = width;
    
    torch::Tensor viewMat = torch::eye(4, device);
    viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
        
    float fovX = 2.0f * std::atan(width / (2.0f * fx));
    float fovY = 2.0f * std::atan(height / (2.0f * fy));

    torch::Tensor projMat = projectionMatrix(0.001f, 1000.0f, fovX, fovY, device);
    torch::Tensor colors =  torch::cat({featuresDc.index({Slice(), None, Slice()}), featuresRest}, 1);

    torch::Tensor conics;
    torch::Tensor depths; 
    torch::Tensor numTilesHit; 
    torch::Tensor cov2d; 
    torch::Tensor camDepths; 
    torch::Tensor rgb;
    torch::Tensor depth; 

    if (device == torch::kCPU){
        auto p = ProjectGaussiansCPU::apply(means, 
                                torch::exp(scales), 
                                1, 
                                quats / quats.norm(2, {-1}, true), 
                                viewMat, 
                                torch::matmul(projMat, viewMat),
                                fx, 
                                fy,
                                cx,
                                cy,
                                height,
                                width);
        xys = p[0];       
        radii = p[1];       
        conics = p[2];      // 椭圆(二次曲线参数)
        cov2d = p[3];       // 协方差矩阵
        camDepths = p[4];   // 相机深度（点投影后相机坐标系下的深度（尺度归一化后的值））
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

        TileBounds tileBounds = std::make_tuple((width + BLOCK_X - 1) / BLOCK_X,
                        (height + BLOCK_Y - 1) / BLOCK_Y,
                        1);
        auto p = ProjectGaussians::apply(means, 
                        torch::exp(scales), 
                        1, 
                        quats / quats.norm(2, {-1}, true), 
                        viewMat, 
                        torch::matmul(projMat, viewMat),
                        fx, 
                        fy,
                        cx,
                        cy,
                        height,
                        width,
                        tileBounds);

        xys = p[0];            // 2D坐标
        depths = p[1];         // 每个高斯的深度
        radii = p[2];          // 半径
        conics = p[3];         // 椭圆(二次曲线参数)
        numTilesHit = p[4];    // 命中数
        #else
            throw std::runtime_error("GPU support not built, use --cpu");
        #endif
    }
    
    if (radii.sum().item<float>() == 0.0f)
        return backgroundColor.repeat({height, width, 1});

    torch::Tensor viewDirs = means.detach() - T.transpose(0, 1).to(device);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    int degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);// 0,1,2,3
    torch::Tensor rgbs;
    
    if (device == torch::kCPU){
        rgbs = SphericalHarmonicsCPU::apply(degreesToUse, viewDirs, colors);
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        rgbs = SphericalHarmonics::apply(degreesToUse, viewDirs, colors);
        #endif
    }
    
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f); 

    if (device == torch::kCPU){
        // todo add depth
        rgb = RasterizeGaussiansCPU::apply(
                xys,
                radii,
                conics,
                rgbs,
                torch::sigmoid(opacities),
                cov2d,
                camDepths,
                height,
                width,
                backgroundColor);
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        if(!depth_flag){
            rgb = RasterizeGaussians::apply( 
                    xys,
                    depths,
                    radii,
                    conics,
                    numTilesHit,
                    rgbs,
                    torch::sigmoid(opacities),
                    height,
                    width,
                    backgroundColor,
                    depth_flag);
            rgb = torch::clamp_max(rgb, 1.0f); 
        }else{
            depth = RasterizeGaussians::apply( 
                xys,
                depths,
                radii,
                conics,
                numTilesHit,
                rgbs,
                torch::sigmoid(opacities),
                height,
                width,
                backgroundColor,
                depth_flag); // 增加深度图像的输出
            depth = depth; // TODO 恢复尺度？
        }
        #endif
    }

    if (depth_flag) {
        return depth;} 
    else {
        return rgb;}
}

void GSMap::optimizersZeroGrad(){
  meansOpt->zero_grad();
  scalesOpt->zero_grad();
  quatsOpt->zero_grad();
  featuresDcOpt->zero_grad();
  featuresRestOpt->zero_grad();
  opacitiesOpt->zero_grad();
}

void GSMap::optimizersStep(){
  meansOpt->step(); 
  scalesOpt->step();
  quatsOpt->step();
  featuresDcOpt->step();
  featuresRestOpt->step();
  opacitiesOpt->step();
}

void GSMap::schedulersStep(int step){
  meansOptScheduler->step(step);
}

int GSMap::getDownscaleFactor(int step){
    return std::pow(2, (std::max<int>)(numDownscales - step / resolutionSchedule, 0)); 
}

void GSMap::addToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples){
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*optimizer->state()[pId]));
    
    std::vector<int64_t> repeats;
    repeats.push_back(nSamples);
    for (long int i = 0; i < paramState->exp_avg().dim() - 1; i++){
        repeats.push_back(1);
    }

    paramState->exp_avg(torch::cat({
        paramState->exp_avg(), 
        torch::zeros_like(paramState->exp_avg().index({idcs.squeeze()})).repeat(repeats)
    }, 0));
    
    paramState->exp_avg_sq(torch::cat({
        paramState->exp_avg_sq(), 
        torch::zeros_like(paramState->exp_avg_sq().index({idcs.squeeze()})).repeat(repeats)
    }, 0));

    optimizer->state().erase(pId);

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif    
    optimizer->state()[newPId] = std::move(paramState);
    optimizer->param_groups()[0].params()[0] = newParam;
}

void GSMap::removeFromOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask){
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*optimizer->state()[pId]));

    paramState->exp_avg(paramState->exp_avg().index({~deletedMask}));
    paramState->exp_avg_sq(paramState->exp_avg_sq().index({~deletedMask}));

    optimizer->state().erase(pId);
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif
    optimizer->param_groups()[0].params()[0] = newParam;
    optimizer->state()[newPId] = std::move(paramState);
}

void GSMap::afterTrain(int step){
    torch::NoGradGuard noGrad;
    std::cout << "After train" << std::endl;
    if (step < stopSplitAt){ // step < 5000
        torch::Tensor visibleMask = (radii > 0).flatten(); 
        torch::Tensor grads = torch::linalg::vector_norm(xys.grad().detach(), 2, { -1 }, false, torch::kFloat32);
        if (!xysGradNorm.numel()){
            xysGradNorm = grads;
            visCounts = torch::ones_like(xysGradNorm);
        }else{
            visCounts.index_put_({visibleMask}, visCounts.index({visibleMask}) + 1);
            xysGradNorm.index_put_({visibleMask}, grads.index({visibleMask}) + xysGradNorm.index({visibleMask}));
        }

        if (!max2DSize.numel()){
            max2DSize = torch::zeros_like(radii, torch::kFloat32);
        }

        torch::Tensor newRadii = radii.detach().index({visibleMask});
        max2DSize.index_put_({visibleMask}, torch::maximum(
                max2DSize.index({visibleMask}), newRadii / static_cast<float>( (std::max)(lastHeight, lastWidth) )
            ));
    }
    std::cout << "()" << std::endl;
    if (step % refineEvery == 0 && step > warmupLength){ // 100/500
        int resetInterval = resetAlphaEvery * refineEvery; // 30*100
        bool doDensification = step < stopSplitAt && step % resetInterval > numCameras + refineEvery;
        torch::Tensor splitsMask;
        const float cullAlphaThresh = 0.1f;

        std::cout << "(0)" << std::endl;
        if (doDensification){
            int numPointsBefore = means.size(0);
            torch::Tensor avgGradNorm = (xysGradNorm / visCounts) * 0.5f * static_cast<float>( (std::max)(lastWidth, lastHeight) );
            torch::Tensor highGrads = (avgGradNorm > densifyGradThresh).squeeze();

            // Split gaussians that are too large
            torch::Tensor splits = (std::get<0>(scales.exp().max(-1)) > densifySizeThresh).squeeze();
            if (step < stopScreenSizeAt){
                splits |= (max2DSize > splitScreenSize).squeeze();
            }
            std::cout << "(00)" << std::endl;
            splits &= highGrads;
            const int nSplitSamples = 2;
            int nSplits = splits.sum().item<int>();

            torch::Tensor centeredSamples = torch::randn({nSplitSamples * nSplits, 3}, device);  // Nx3 of axis-aligned scales
            torch::Tensor scaledSamples = torch::exp(scales.index({splits}).repeat({nSplitSamples, 1})) * centeredSamples;
            torch::Tensor qs = quats.index({splits}) / torch::linalg::vector_norm(quats.index({splits}), 2, { -1 }, true, torch::kFloat32);
            torch::Tensor rots = quatToRotMat(qs.repeat({nSplitSamples, 1}));
            torch::Tensor rotatedSamples = torch::bmm(rots, scaledSamples.index({"...", None})).squeeze();
            torch::Tensor splitMeans = rotatedSamples + means.index({splits}).repeat({nSplitSamples, 1});
            
            torch::Tensor splitFeaturesDc = featuresDc.index({splits}).repeat({nSplitSamples, 1});
            torch::Tensor splitFeaturesRest = featuresRest.index({splits}).repeat({nSplitSamples, 1, 1});
            
            torch::Tensor splitOpacities = opacities.index({splits}).repeat({nSplitSamples, 1});

            const float sizeFac = 1.6f;
            torch::Tensor splitScales = torch::log(torch::exp(scales.index({splits})) / sizeFac).repeat({nSplitSamples, 1});
            scales.index({splits}) = torch::log(torch::exp(scales.index({splits})) / sizeFac);
            torch::Tensor splitQuats = quats.index({splits}).repeat({nSplitSamples, 1});

            // Duplicate gaussians that are too small
            torch::Tensor dups = (std::get<0>(scales.exp().max(-1)) <= densifySizeThresh).squeeze();
            dups &= highGrads;
            torch::Tensor dupMeans = means.index({dups});
            torch::Tensor dupFeaturesDc = featuresDc.index({dups});
            torch::Tensor dupFeaturesRest = featuresRest.index({dups});
            torch::Tensor dupOpacities = opacities.index({dups});
            torch::Tensor dupScales = scales.index({dups});
            torch::Tensor dupQuats = quats.index({dups});

            means = torch::cat({means.detach(), splitMeans, dupMeans}, 0).requires_grad_(); 
            featuresDc = torch::cat({featuresDc.detach(), splitFeaturesDc, dupFeaturesDc}, 0).requires_grad_();
            featuresRest = torch::cat({featuresRest.detach(), splitFeaturesRest, dupFeaturesRest}, 0).requires_grad_();
            opacities = torch::cat({opacities.detach(), splitOpacities, dupOpacities}, 0).requires_grad_();
            scales = torch::cat({scales.detach(), splitScales, dupScales}, 0).requires_grad_();
            quats = torch::cat({quats.detach(), splitQuats, dupQuats}, 0).requires_grad_();

            max2DSize = torch::cat({
                max2DSize,
                torch::zeros_like(splitScales.index({Slice(), 0})),
                torch::zeros_like(dupScales.index({Slice(), 0}))
            }, 0);

            torch::Tensor splitIdcs = torch::where(splits)[0];

            addToOptimizer(meansOpt, means, splitIdcs, nSplitSamples);
            addToOptimizer(scalesOpt, scales, splitIdcs, nSplitSamples);
            addToOptimizer(quatsOpt, quats, splitIdcs, nSplitSamples);
            addToOptimizer(featuresDcOpt, featuresDc, splitIdcs, nSplitSamples);
            addToOptimizer(featuresRestOpt, featuresRest, splitIdcs, nSplitSamples);
            addToOptimizer(opacitiesOpt, opacities, splitIdcs, nSplitSamples);
            
            torch::Tensor dupIdcs = torch::where(dups)[0];
            addToOptimizer(meansOpt, means, dupIdcs, 1);
            addToOptimizer(scalesOpt, scales, dupIdcs, 1);
            addToOptimizer(quatsOpt, quats, dupIdcs, 1);
            addToOptimizer(featuresDcOpt, featuresDc, dupIdcs, 1);
            addToOptimizer(featuresRestOpt, featuresRest, dupIdcs, 1);
            addToOptimizer(opacitiesOpt, opacities, dupIdcs, 1);

            splitsMask = torch::cat({
                splits,
                torch::full({nSplitSamples * splits.sum().item<int>() + dups.sum().item<int>()}, false, torch::TensorOptions().dtype(torch::kBool).device(device))
            }, 0);

        }

        if (doDensification){
            // Cull
            int numPointsBefore = means.size(0);

            torch::Tensor culls = (torch::sigmoid(opacities) < cullAlphaThresh).squeeze();
            if (splitsMask.numel()){
                culls |= splitsMask;
            }

            if (step > refineEvery * resetAlphaEvery){
                const float cullScaleThresh = 0.5f; // cull huge gaussians
                const float cullScreenSize = 0.15f; // % of screen space
                torch::Tensor huge = std::get<0>(torch::exp(scales).max(-1)) > cullScaleThresh;
                if (step < stopScreenSizeAt){
                    huge |= max2DSize > cullScreenSize;
                }
                culls |= huge;
            }

            int cullCount = torch::sum(culls).item<int>();
            if (cullCount > 0){
                means = means.index({~culls}).detach().requires_grad_(); 
                scales = scales.index({~culls}).detach().requires_grad_();
                quats = quats.index({~culls}).detach().requires_grad_();
                featuresDc = featuresDc.index({~culls}).detach().requires_grad_();
                featuresRest = featuresRest.index({~culls}).detach().requires_grad_();
                opacities = opacities.index({~culls}).detach().requires_grad_();

                removeFromOptimizer(meansOpt, means, culls);
                removeFromOptimizer(scalesOpt, scales, culls);
                removeFromOptimizer(quatsOpt, quats, culls);
                removeFromOptimizer(featuresDcOpt, featuresDc, culls);
                removeFromOptimizer(featuresRestOpt, featuresRest, culls);
                removeFromOptimizer(opacitiesOpt, opacities, culls);
                
            }
        }
        if (step < stopSplitAt && step % resetInterval == refineEvery){
            float resetValue = cullAlphaThresh * 2.0f;
            opacities = torch::clamp_max(opacities, torch::logit(torch::tensor(resetValue)).item<float>());

            // Reset optimizer
            torch::Tensor param = opacitiesOpt->param_groups()[0].params()[0];
            #if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
                auto pId = param.unsafeGetTensorImpl();
            #else
                auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
            #endif    
            auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*opacitiesOpt->state()[pId]));
            paramState->exp_avg(torch::zeros_like(paramState->exp_avg()));
            paramState->exp_avg_sq(torch::zeros_like(paramState->exp_avg_sq()));

        }
        // Clear
        xysGradNorm = torch::Tensor();
        visCounts = torch::Tensor();
        max2DSize = torch::Tensor();

        if (device != torch::kCPU){
            #ifdef USE_HIP
                    c10::hip::HIPCachingAllocator::emptyCache();
            #elif defined(USE_CUDA)
                    c10::cuda::CUDACachingAllocator::emptyCache();
            #endif
        }
    }
}

void GSMap::save(const std::string &filename){
    if (fs::path(filename).extension().string() == ".splat"){
        saveSplat(filename);
    }else{
        savePly(filename);
    }
    std::cout << "Wrote " << filename << std::endl;
}

void GSMap::savePly(const std::string &filename){
    std::ofstream o(filename, std::ios::binary);
    int numPoints = means.size(0);

    o << "ply" << std::endl;
    o << "format binary_little_endian 1.0" << std::endl;
    o << "comment Generated by opensplat" << std::endl;
    o << "element vertex " << numPoints << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;
    o << "property float nx" << std::endl;
    o << "property float ny" << std::endl;
    o << "property float nz" << std::endl;

    for (int i = 0; i < featuresDc.size(1); i++){
        o << "property float f_dc_" << i << std::endl;
    }

    // Match Inria's version
    torch::Tensor featuresRestCpu = featuresRest.cpu().transpose(1, 2).reshape({numPoints, -1});
    for (int i = 0; i < featuresRestCpu.size(1); i++){
        o << "property float f_rest_" << i << std::endl;
    }

    o << "property float opacity" << std::endl;

    o << "property float scale_0" << std::endl;
    o << "property float scale_1" << std::endl;
    o << "property float scale_2" << std::endl;

    o << "property float rot_0" << std::endl;
    o << "property float rot_1" << std::endl;
    o << "property float rot_2" << std::endl;
    o << "property float rot_3" << std::endl;
    
    o << "end_header" << std::endl;

    float zeros[] = { 0.0f, 0.0f, 0.0f };

    torch::Tensor meansCpu = means.cpu(); //
    torch::Tensor featuresDcCpu = featuresDc.cpu();
    torch::Tensor opacitiesCpu = opacities.cpu();
    torch::Tensor scalesCpu = scales.cpu();
    torch::Tensor quatsCpu = quats.cpu(); 

    for (size_t i = 0; i < numPoints; i++) {
        o.write(reinterpret_cast<const char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(zeros), sizeof(float) * 3); // TODO: do we need to write zero normals?
        o.write(reinterpret_cast<const char *>(featuresDcCpu[i].data_ptr()), sizeof(float) * featuresDcCpu.size(1));
        o.write(reinterpret_cast<const char *>(featuresRestCpu[i].data_ptr()), sizeof(float) * featuresRestCpu.size(1));
        o.write(reinterpret_cast<const char *>(opacitiesCpu[i].data_ptr()), sizeof(float) * 1);
        o.write(reinterpret_cast<const char *>(scalesCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(quatsCpu[i].data_ptr()), sizeof(float) * 4);
    }

    o.close();
}

void GSMap::savemydebugPly(const std::string &filename){
    std::ofstream o(filename, std::ios::binary);
    int numPoints = means.size(0);

    o << "ply" << std::endl;
    o << "format ascii 1.0" << std::endl;
    o << "comment Generated by opensplat" << std::endl;
    o << "element vertex " << numPoints << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;
    o << "property float nx" << std::endl;
    o << "property float ny" << std::endl;
    o << "property float nz" << std::endl;

    for (int i = 0; i < featuresDc.size(1); i++){
        o << "property float f_dc_" << i << std::endl;
    }

    // Match Inria's version
    torch::Tensor featuresRestCpu = featuresRest.cpu().transpose(1, 2).reshape({numPoints, -1});
    for (int i = 0; i < featuresRestCpu.size(1); i++){
        o << "property float f_rest_" << i << std::endl;
    }

    o << "property float opacity" << std::endl;

    o << "property float scale_0" << std::endl;
    o << "property float scale_1" << std::endl;
    o << "property float scale_2" << std::endl;

    o << "property float rot_0" << std::endl;
    o << "property float rot_1" << std::endl;
    o << "property float rot_2" << std::endl;
    o << "property float rot_3" << std::endl;
    
    o << "end_header" << std::endl;

    float zeros[] = { 0.0f, 0.0f, 0.0f };

    torch::Tensor meansCpu = means.cpu();
    torch::Tensor featuresDcCpu = featuresDc.cpu();
    torch::Tensor opacitiesCpu = opacities.cpu();
    torch::Tensor scalesCpu = scales.cpu();
    torch::Tensor quatsCpu = quats.cpu();

    for (size_t i = 0; i < numPoints; i++) {
        o << meansCpu[i][0].item<float>() << " " << meansCpu[i][1].item<float>() << " " << meansCpu[i][2].item<float>() << " ";
        o << "0.0 0.0 0.0 "; // TODO: do we need to write zero normals?
        for (int j = 0; j < featuresDcCpu.size(1); j++) {
            o << featuresDcCpu[i][j].item<float>() << " ";
        }
        for (int j = 0; j < featuresRestCpu.size(1); j++) {
            o << featuresRestCpu[i][j].item<float>() << " ";
        }
        o << opacitiesCpu[i].item<float>() << " ";
        o << scalesCpu[i][0].item<float>() << " " << scalesCpu[i][1].item<float>() << " " << scalesCpu[i][2].item<float>() << " ";
        o << quatsCpu[i][0].item<float>() << " " << quatsCpu[i][1].item<float>() << " " << quatsCpu[i][2].item<float>() << " " << quatsCpu[i][3].item<float>() << std::endl;
    }    

    o.close();
    std::cout << "Wrote " << filename << std::endl;
}

void GSMap::saveSplat(const std::string &filename){
    std::ofstream o(filename, std::ios::binary);
    int numPoints = means.size(0);

    torch::Tensor meansCpu = means.cpu();
    torch::Tensor scalesCpu = torch::exp(scales.cpu());
    torch::Tensor rgbsCpu = (sh2rgb(featuresDc.cpu()) * 255.0f).toType(torch::kUInt8);
    torch::Tensor opac = (1.0f + torch::exp(-opacities.cpu()));
    torch::Tensor opacitiesCpu = torch::clamp(((1.0f / opac) * 255.0f), 0.0f, 255.0f).toType(torch::kUInt8);
    torch::Tensor quatsCpu = torch::clamp(quats.cpu() * 128.0f + 128.0f, 0.0f, 255.0f).toType(torch::kUInt8);

    std::vector< size_t > splatIndices( numPoints );
    std::iota( splatIndices.begin(), splatIndices.end(), 0 );
    torch::Tensor order = (scalesCpu.index({"...", 0}) + 
                            scalesCpu.index({"...", 1}) + 
                            scalesCpu.index({"...", 2})) / 
                            opac.index({"...", 0});
    float *orderPtr = reinterpret_cast<float *>(order.data_ptr());

    std::sort(splatIndices.begin(), splatIndices.end(), 
        [&orderPtr](size_t const &a, size_t const &b) {
            return orderPtr[a] > orderPtr[b];
        });

    for (int i = 0; i < numPoints; i++){
        size_t idx = splatIndices[i];

        o.write(reinterpret_cast<const char *>(meansCpu[idx].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(scalesCpu[idx].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(rgbsCpu[idx].data_ptr()), sizeof(uint8_t) * 3);
        o.write(reinterpret_cast<const char *>(opacitiesCpu[idx].data_ptr()), sizeof(uint8_t) * 1);
        o.write(reinterpret_cast<const char *>(quatsCpu[idx].data_ptr()), sizeof(uint8_t) * 4);
    }
    o.close();
}

void GSMap::saveDebugPly(const std::string &filename){
    // A standard PLY
    std::ofstream o(filename, std::ios::binary);
    int numPoints = means.size(0);

    o << "ply" << std::endl;
    o << "format binary_little_endian 1.0" << std::endl;
    o << "comment Generated by opensplat" << std::endl;
    o << "element vertex " << numPoints << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;
    o << "property uchar red" << std::endl;
    o << "property uchar green" << std::endl;
    o << "property uchar blue" << std::endl;
    o << "end_header" << std::endl;

    torch::Tensor meansCpu = means.cpu(); 
    torch::Tensor rgbsCpu = (sh2rgb(featuresDc.cpu()) * 255.0f).toType(torch::kUInt8);

    for (size_t i = 0; i < numPoints; i++) {
        o.write(reinterpret_cast<const char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(rgbsCpu[i].data_ptr()), sizeof(uint8_t) * 3);
    }

    o.close();
    std::cout << "Wrote " << filename << std::endl;
}

torch::Tensor GSMap::mainLoss(torch::Tensor &rgb, torch::Tensor &gt, float ssimWeight){
    torch::Tensor ssimLoss = 1.0f - ssim.eval(rgb, gt);
    torch::Tensor l1Loss = l1(rgb, gt);
    return (1.0f - ssimWeight) * l1Loss + ssimWeight * ssimLoss;
}

void GSMap::processCamera(Camera &cam){
    torch::Tensor depth = forward(cam, 10000, true);
    {
	std::lock_guard<std::mutex> lock(depthQueueMutex);
	depthQueue.push(depth);
	depthConditionVar.notify_one(); 
	}
    return ;
}

void GSMap::run(){
    while(true){
        std::unique_lock<std::mutex> lock(camQueueMutex);
        conditionVar.wait(lock, [this] { return !cameraQueue.empty();});
        Camera camera = cameraQueue.front();
        cameraQueue.pop();
        lock.unlock();
        processCamera(camera); 
    }
}