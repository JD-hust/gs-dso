#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/version.h>
#include "nerfstudio.hpp"
#include "kdtree_tensor.hpp"
#include "spherical_harmonics.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include "optim_scheduler.hpp"

using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor randomQuatTensor(long long n);
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);
torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt);
torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt);

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval, 
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        const torch::Device &device) :
    numCameras(numCameras),
    numDownscales(numDownscales), resolutionSchedule(resolutionSchedule), shDegree(shDegree), shDegreeInterval(shDegreeInterval), 
    refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery), stopSplitAt(maxSteps / 2), densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh), stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
    maxSteps(maxSteps), keepCrs(keepCrs),
    device(device), ssim(11, 3){
    
    long long numPoints = inputData.points.xyz.size(0); 
    scale = inputData.scale; 
    translation = inputData.translation;

    torch::manual_seed(42); // 设置随机数种子

    means = inputData.points.xyz.to(device).requires_grad_();
    scales = PointsTensor(inputData.points.xyz).scales().repeat({1, 3}).log().to(device).requires_grad_();
    quats = randomQuatTensor(numPoints).to(device).requires_grad_();

    int dimSh = numShBases(shDegree); 
    torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(inputData.points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32);
    shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;
    featuresDc = shs.index({Slice(), 0, Slice()}).to(device).requires_grad_(); 
    featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(device).requires_grad_();

    opacities = torch::logit(0.8f * torch::ones({numPoints, 1})).to(device).requires_grad_(); 
    
    backgroundColor = torch::tensor({0.0f, 0.0f, 0.0f}, device);
    meansOpt = new torch::optim::Adam({means}, torch::optim::AdamOptions(0.0001)); //0.00016
    scalesOpt = new torch::optim::Adam({scales}, torch::optim::AdamOptions(0.005));
    quatsOpt = new torch::optim::Adam({quats}, torch::optim::AdamOptions(0.001));
    featuresDcOpt = new torch::optim::Adam({featuresDc}, torch::optim::AdamOptions(0.025)); //0.00125
    featuresRestOpt = new torch::optim::Adam({featuresRest}, torch::optim::AdamOptions(0.000125));
    opacitiesOpt = new torch::optim::Adam({opacities}, torch::optim::AdamOptions(0.01)); //0.05

    meansOptScheduler = new OptimScheduler(meansOpt, 0.0000016f, maxSteps); // 设置位置的学习率调度器
  }
  ~Model(){
    // 
    delete meansOpt;
    delete scalesOpt;
    delete quatsOpt;
    delete featuresDcOpt;
    delete featuresRestOpt;
    delete opacitiesOpt;

    delete meansOptScheduler;
  }

  // 成员函数，默认为public
  torch::Tensor forward(Camera& cam, int step, bool depth_flag); // before torch::Tensor forward(Camera& cam, int step);
  void optimizersZeroGrad();
  void optimizersStep();
  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void save(const std::string &filename);
  void savePly(const std::string &filename);
  void saveSplat(const std::string &filename);
  void saveDebugPly(const std::string &filename);
  void savemydebugPly(const std::string &filename);
  void saveConfig(const std::string &filename);
  torch::Tensor mainLoss(torch::Tensor &rgb, torch::Tensor &gt, float ssimWeight);

  void addToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples);
  void removeFromOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask);

  torch::Tensor means;       // 高斯的位置
  torch::Tensor scales;      // 高斯的缩放因子
  torch::Tensor quats;       // 高斯的四元数
  torch::Tensor featuresDc;  // 高斯的第一维颜色
  torch::Tensor featuresRest;// 高斯的其他维颜色
  torch::Tensor opacities;   // 高斯的透明度

  torch::optim::Adam *meansOpt;
  torch::optim::Adam *scalesOpt;
  torch::optim::Adam *quatsOpt;
  torch::optim::Adam *featuresDcOpt;
  torch::optim::Adam *featuresRestOpt;
  torch::optim::Adam *opacitiesOpt;

  OptimScheduler *meansOptScheduler;

  torch::Tensor radii; // set in forward()
  torch::Tensor xys; // set in forward()
  int lastHeight; // set in forward()
  int lastWidth; // set in forward()

  torch::Tensor xysGradNorm; // set in afterTrain()
  torch::Tensor visCounts; // set in afterTrain()  
  torch::Tensor max2DSize; // set in afterTrain()

  torch::Tensor backgroundColor;
  torch::Device device;
  SSIM ssim;

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs;

  float scale;
  torch::Tensor translation;
};

struct Model_gmm : public Model{
  Model_gmm(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval, 
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        const torch::Device &device) :
    Model(inputData, numCameras, numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
          refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
          maxSteps, keepCrs, device) {
      
      long long numPoints = inputData.points.xyz.size(0); 
      scale = inputData.scale; 
      translation = inputData.translation;


      torch::manual_seed(42); 

      means = inputData.points.xyz.to(device).requires_grad_();
      scales = PointsTensor(inputData.points.xyz).scales().repeat({1, 3}).log().to(device).requires_grad_();
      quats = randomQuatTensor(numPoints).to(device).requires_grad_();

      int dimSh = numShBases(shDegree); 
      torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
      shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(inputData.points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32);
      shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;
      featuresDc = shs.index({Slice(), 0, Slice()}).to(device).requires_grad_(); 
      featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(device).requires_grad_();

      opacities = torch::logit(0.8f * torch::ones({numPoints, 1})).to(device).requires_grad_(); // 0.1f
      
      backgroundColor = torch::tensor({0.0f, 0.0f, 0.0f}, device); 
      meansOpt = new torch::optim::Adam({means}, torch::optim::AdamOptions(0.0001)); //0.00016
      scalesOpt = new torch::optim::Adam({scales}, torch::optim::AdamOptions(0.005));
      quatsOpt = new torch::optim::Adam({quats}, torch::optim::AdamOptions(0.001));
      featuresDcOpt = new torch::optim::Adam({featuresDc}, torch::optim::AdamOptions(0.025)); //0.00125
      featuresRestOpt = new torch::optim::Adam({featuresRest}, torch::optim::AdamOptions(0.000125));
      opacitiesOpt = new torch::optim::Adam({opacities}, torch::optim::AdamOptions(0.01)); //0.05

      meansOptScheduler = new OptimScheduler(meansOpt, 0.0000016f, maxSteps); 
  }

};

#endif