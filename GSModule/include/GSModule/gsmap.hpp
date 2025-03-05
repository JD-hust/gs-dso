#ifndef GSMAP_H
#define GSMAP_H

#include <iostream>
#include <queue>
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/version.h>
//#include "nerfstudio.hpp"
//#include "kdtree_tensor.hpp"
#include "spherical_harmonics.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include "optim_scheduler.hpp"

using namespace torch::indexing;
using namespace torch::autograd;

//torch::Tensor randomQuatTensor(long long n);
//torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);
//torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt);
//torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt);
// 修改为：读取预训练的模型
struct GSMap{
  GSMap(const std::string &inputPLY, const std::string &inputConfig){


    Config config = inputConfigFromX(inputConfig);

    numDownscales = config.numDownscales; 
    resolutionSchedule = config.resolutionSchedule; 
    shDegree = config.shDegree; 
    shDegreeInterval = config.shDegreeInterval; 
    refineEvery = config.refineEvery; 
    warmupLength = config.warmupLength; 
    resetAlphaEvery = config.resetAlphaEvery; 
    densifyGradThresh = config.densifyGradThresh; 
    densifySizeThresh = config.densifySizeThresh; 
    stopScreenSizeAt = config.stopScreenSizeAt;
    splitScreenSize = config.splitScreenSize; 
    maxSteps = config.maxSteps; 
    keepCrs = config.keepCrs; 
    device = config.device;
    
    InputGS inputGS = inputGSFromX(inputPLY, config);

    long long numPoints = inputGS.means.size(0);

    means = inputGS.means.to(device);
    scales = inputGS.scales.to(device);
    quats = inputGS.quats.to(device);
    opacities = inputGS.opacities.to(device);
    featuresDc = inputGS.featuresDc.to(device);
    featuresRest = inputGS.featuresRest.to(device);
    backgroundColor = torch::tensor({240,248,255}, device) / 255.0f; 

    meansOpt = new torch::optim::Adam({means}, torch::optim::AdamOptions(0.00016));
    scalesOpt = new torch::optim::Adam({scales}, torch::optim::AdamOptions(0.005));
    quatsOpt = new torch::optim::Adam({quats}, torch::optim::AdamOptions(0.001));
    featuresDcOpt = new torch::optim::Adam({featuresDc}, torch::optim::AdamOptions(0.0025));
    featuresRestOpt = new torch::optim::Adam({featuresRest}, torch::optim::AdamOptions(0.000125));
    opacitiesOpt = new torch::optim::Adam({opacities}, torch::optim::AdamOptions(0.05));

    meansOptScheduler = new OptimScheduler(meansOpt, 0.0000016f, maxSteps); 

    cameraQueue = std::queue<Camera>();
    depthQueue = std::queue<torch::Tensor>();

  }

  ~GSMap(){
    // 
    delete meansOpt;
    delete scalesOpt;
    delete quatsOpt;
    delete featuresDcOpt;
    delete featuresRestOpt;
    delete opacitiesOpt;

    delete meansOptScheduler;
  }

  torch::Tensor forward(Camera& cam, int step, bool depth_flag);
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
  torch::Tensor mainLoss(torch::Tensor &rgb, torch::Tensor &gt, float ssimWeight);

  void addToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples);
  void removeFromOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask);

  void processCamera(Camera &cam);
  void run();

  std::queue<Camera> cameraQueue;
  std::queue<torch::Tensor> depthQueue;

  std::mutex camQueueMutex, depthQueueMutex;
  std::condition_variable conditionVar, depthConditionVar;

  // 训练参数
  torch::Tensor means;       // 高斯的位置
  torch::Tensor scales;      // 高斯的缩放因子
  torch::Tensor quats;       // 高斯的四元数
  torch::Tensor featuresDc;  // 高斯的第一维颜色
  torch::Tensor featuresRest;// 高斯的其他维颜色
  torch::Tensor opacities;   // 高斯的透明度
  // 优化器
  torch::optim::Adam *meansOpt;
  torch::optim::Adam *scalesOpt;
  torch::optim::Adam *quatsOpt;
  torch::optim::Adam *featuresDcOpt;
  torch::optim::Adam *featuresRestOpt;
  torch::optim::Adam *opacitiesOpt;
  // 优化器调度器
  OptimScheduler *meansOptScheduler;

  torch::Tensor radii; // set in forward()
  torch::Tensor xys; // set in forward()
  int lastHeight; // set in forward()
  int lastWidth; // set in forward()

  torch::Tensor xysGradNorm; // set in afterTrain()
  torch::Tensor visCounts; // set in afterTrain()  
  torch::Tensor max2DSize; // set in afterTrain()

  // 其他:背景颜色、设备、SSIM
  torch::Tensor backgroundColor;
  torch::Device device = torch::kCUDA;
  SSIM ssim = SSIM(11, 3);

  int numCameras = 0;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval = 0;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs = false;

  //float scale;
  //torch::Tensor translation;
};


#endif