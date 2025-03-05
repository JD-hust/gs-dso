#include <filesystem>
#include <nlohmann/json.hpp>
#include "input_data.hpp"
#include "cv_utils.hpp"
#include <opencv2/opencv.hpp>
namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot); }
namespace osfm { InputData inputDataFromOpenSfM(const std::string &projectRoot); }

InputData inputDataFromX(const std::string &projectRoot){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot);
    }else if (fs::exists(root / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM(projectRoot);
    }else if (fs::exists(root / "opensfm" / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM((root / "opensfm").string());
    }else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio project folder)");
    }
}

torch::Tensor Camera::getIntrinsicsMatrix(){
    return torch::tensor({{fx, 0.0f, cx},
                          {0.0f, fy, cy},
                          {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
}

void Camera::loadImage(float downscaleFactor){
    // Populates image and K, then updates the camera parameters
    // Caution: this function has destructive behaviors
    // and should be called only once
    if (image.numel()) std::runtime_error("loadImage already called");
    std::cout << "Loading " << filePath << std::endl;

    float scaleFactor = 1.0f / downscaleFactor;
    cv::Mat cImg = imreadRGB(filePath);
    
    // 调试，检查可视化图像
    // cv::imshow("image", cImg);
    // cv::waitKey(0);

    float rescaleF = 1.0f;
    // If camera intrinsics don't match the image dimensions 
    if (cImg.rows != height || cImg.cols != width){
        rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(height);
    }
    fx *= scaleFactor * rescaleF;
    fy *= scaleFactor * rescaleF;
    cx *= scaleFactor * rescaleF;
    cy *= scaleFactor * rescaleF;

    if (downscaleFactor > 1.0f){
        float f = 1.0f / downscaleFactor;
        cv::resize(cImg, cImg, cv::Size(), f, f, cv::INTER_AREA);
    }

    K = getIntrinsicsMatrix();
    cv::Rect roi;

    if (hasDistortionParameters()){
        // Undistort
        std::vector<float> distCoeffs = undistortionParameters();
        cv::Mat cK = floatNxNtensorToMat(K);
        cv::Mat newK = cv::getOptimalNewCameraMatrix(cK, distCoeffs, cv::Size(cImg.cols, cImg.rows), 0, cv::Size(), &roi);

        cv::Mat undistorted = cv::Mat::zeros(cImg.rows, cImg.cols, cImg.type());
        cv::undistort(cImg, undistorted, cK, distCoeffs, newK);
        
        image = imageToTensor(undistorted);
        K = floatNxNMatToTensor(newK);
    }else{
        roi = cv::Rect(0, 0, cImg.cols, cImg.rows);
        image = imageToTensor(cImg);
    }

    // Crop to ROI
    image = image.index({Slice(roi.y, roi.y + roi.height), Slice(roi.x, roi.x + roi.width), Slice()});

    // Update parameters
    height = image.size(0);
    width = image.size(1);
    fx = K[0][0].item<float>();
    fy = K[1][1].item<float>();
    cx = K[0][2].item<float>();
    cy = K[1][2].item<float>();
}

torch::Tensor Camera::getImage(int downscaleFactor){
    if (downscaleFactor <= 1) return image;
    else{

        // torch::jit::script::Module container = torch::jit::load("gt.pt");
        // return container.attr("val").toTensor();

        if (imagePyramids.find(downscaleFactor) != imagePyramids.end()){
            return imagePyramids[downscaleFactor];
        }

        // Rescale, store and return
        cv::Mat cImg = tensorToImage(image);
        cv::resize(cImg, cImg, cv::Size(cImg.cols / downscaleFactor, cImg.rows / downscaleFactor), 0.0, 0.0, cv::INTER_AREA);
        torch::Tensor t = imageToTensor(cImg);
        imagePyramids[downscaleFactor] = t;
        return t;
    }
}

bool Camera::hasDistortionParameters(){
    return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
}

std::vector<float> Camera::undistortionParameters(){
    std::vector<float> p = { k1, k2, p1, p2, k3, 0.0f, 0.0f, 0.0f };
    return p;
}

std::tuple<std::vector<Camera>, Camera *> InputData::getCameras(bool validate, const std::string &valImage){
    if (!validate) return std::make_tuple(cameras, nullptr); // 如果不验证，则返回所有相机
    else{
        size_t valIdx = -1;
        std::srand(42);

        if (valImage == "random"){
            valIdx = std::rand() % cameras.size();
        }else{ // 如果指定验证相机的文件名，则返回指定相机
            for (size_t i = 0; i < cameras.size(); i++){
                if (fs::path(cameras[i].filePath).filename().string() == valImage){
                    valIdx = i;
                    break;
                }
            }
            if (valIdx == -1) throw std::runtime_error(valImage + " not in the list of cameras");
        }

        std::vector<Camera> cams;
        Camera *valCam = nullptr;

        // 指针指向验证相机，其余相机存入cams
        for (size_t i = 0; i < cameras.size(); i++){
            if (i != valIdx) cams.push_back(cameras[i]);
            else valCam = &cameras[i]; 
        }

        return std::make_tuple(cams, valCam);
    }
}


void InputData::saveCameras(const std::string &filename, bool keepCrs){
    json j = json::array();
    
    for (size_t i = 0; i < cameras.size(); i++){
        Camera &cam = cameras[i];

        json camera = json::object();
        camera["id"] = i;
        camera["img_name"] = fs::path(cam.filePath).filename().string();
        camera["width"] = cam.width;
        camera["height"] = cam.height;
        camera["fx"] = cam.fx;
        camera["fy"] = cam.fy;

        torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)}).squeeze();
        
        // Flip z and y
        R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f})));

        if (keepCrs) T = (T / scale) + translation;

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++) {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++) {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
        j.push_back(camera);
    }
    
    std::ofstream of(filename);
    of << j;
    of.close();

    std::cout << "Wrote " << filename << std::endl;
}

inline int numShbases(int degree){
    switch(degree){
        case 0:
            return 1;
        case 1:
            return 4;
        case 2:
            return 9;
        case 3:
            return 16;
        default:
            return 25;
    }
}

//  todo: 支持不同阶数的sh
InputGS inputGSFromX(const std::string &inputPLY, const Config &inputConfig){
    InputGS gs;
    int numPoints = 0;
    int numshs = 0;
    fs::path ply(inputPLY);
    if (!fs::exists(ply)) throw std::runtime_error("Invalid input PLY file");
    
    std::ifstream in(ply.string());
    std::string line;
    std::string subline;
    std::getline(in, line); // ply
    std::getline(in, line); // format ascii 1.0
    if (line == "format ascii 1.0"){  
        // Load the PLY file: ascii 1.0 格式
        std::getline(in, line); // comment Generated by opensplat
        std::getline(in, line); // element vertex 1234
        numPoints = std::stoi(line.substr(15));
        std::getline(in, line); // property float x
        std::getline(in, line); // property float y
        std::getline(in, line); // property float z
        std::getline(in, line); // property float nx
        std::getline(in, line); // property float ny
        std::getline(in, line); // property float nz
        std::getline(in, line); // property float f_dc_0
        std::getline(in, line); // property float f_dc_1
        std::getline(in, line); // property float f_dc_2
        std::getline(in, line); // property float f_rest_0
        // 判断line是否由property float f_rest_开头
        if (line == "property float f_rest_0"){
            subline = line.substr(0, 22);
            while(subline == "property float f_rest_"){
                numshs++; // 计算sh的数量,暂时无用
                std::getline(in, line); // property float opacity
                subline = line.substr(0, 22);
            }
        }
        std::getline(in, line); // property float scale_0
        std::getline(in, line); // property float scale_1
        std::getline(in, line); // property float scale_2
        std::getline(in, line); // property float rot_0
        std::getline(in, line); // property float rot_1
        std::getline(in, line); // property float rot_2
        std::getline(in, line); // property float rot_3
        std::getline(in, line); // end_header

        // gs初始化
        gs.means = torch::zeros({numPoints, 3});
        gs.scales = torch::zeros({numPoints, 3});
        gs.quats = torch::zeros({numPoints, 4});
        gs.opacities = torch::logit(0.1f * torch::ones({numPoints, 1}));

        // todo 阶数的设置
        int dimSh = numShbases(inputConfig.shDegree); // 阶数
        //gs.featuresDc = shs.index({Slice(), 0, Slice()}).to(inputConfig.device).requires_grad_(); 
        //gs.featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(inputConfig.device).requires_grad_();
        gs.featuresDc = torch::zeros({numPoints, 3});
        gs.featuresRest = torch::zeros({numPoints, 3*(dimSh-1)});
        // 读取数据
        for (int i = 0; i < numPoints; i++){
            float x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, f_rest_0, f_rest_1, f_rest_2, f_rest_3, f_rest_4, f_rest_5, f_rest_6, f_rest_7, f_rest_8, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3;
            in >> x >> y >> z >> nx >> ny >> nz >> f_dc_0 >> f_dc_1 >> f_dc_2 >> f_rest_0 >> f_rest_1 >> f_rest_2 >> f_rest_3 >> f_rest_4 >> f_rest_5 >> f_rest_6 >> f_rest_7 >> f_rest_8 >> opacity >> scale_0 >> scale_1 >> scale_2 >> rot_0 >> rot_1 >> rot_2 >> rot_3;
            gs.means[i][0] = x;
            gs.means[i][1] = y;
            gs.means[i][2] = z;
            gs.opacities[i][0] = opacity;
            gs.featuresDc[i][0] = f_dc_0;
            gs.featuresDc[i][0] = f_dc_1;
            gs.featuresDc[i][2] = f_dc_2;
            gs.featuresRest[i][0] = f_rest_0;
            gs.featuresRest[i][1] = f_rest_1;
            gs.featuresRest[i][2] = f_rest_2;
            gs.featuresRest[i][3] = f_rest_3;
            gs.featuresRest[i][4] = f_rest_4;
            gs.featuresRest[i][5] = f_rest_5;
            gs.featuresRest[i][6] = f_rest_6;
            gs.featuresRest[i][7] = f_rest_7;
            gs.featuresRest[i][8] = f_rest_8;
            gs.scales[i][0] = scale_0;
            gs.scales[i][1] = scale_1;
            gs.scales[i][2] = scale_2;
            gs.quats[i][0] = rot_0;
            gs.quats[i][1] = rot_1;
            gs.quats[i][2] = rot_2;
            gs.quats[i][3] = rot_3;
        }  
        gs.featuresRest = gs.featuresRest.reshape({numPoints, 3, dimSh-1}).transpose(1, 2); // 重塑张量

    }
    else if (line == "format binary_little_endian 1.0")
    {
        throw std::runtime_error("PLY file must be in ascii format");
    }
    /*
    std::ifstream in(ply.string(), std::ios::binary);
    std::string line;
    std::getline(in, line); // ply
    std::getline(in, line); // format binary_little_endian 1.0
    std::getline(in, line); // comment Generated by opensplat
    std::getline(in, line); // element vertex 1234
    std::getline(in, line); // property float x
    std::getline(in, line); // property float y
    std::getline(in, line); // property float z
    std::getline(in, line); // property float nx
    std::getline(in, line); // property float ny
    std::getline(in, line); // property float nz

    int numPoints = std::stoi(line.substr(15));
    std::getline(in, line); // end_header

    // Read the data
    torch::Tensor means = torch::zeros({numPoints, 3});
    torch::Tensor scales = torch::zeros({numPoints, 3});
    torch::Tensor quats = torch::zeros({numPoints, 4});
    torch::Tensor featuresDc = torch::zeros({numPoints, 3});
    torch::Tensor featuresRest = torch::zeros({numPoints, 9});
    torch::Tensor opacities = torch::logit(0.1f * torch::ones({numPoints, 1}));

    for (int i = 0; i < numPoints; i++){
        float x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, f_rest_0, f_rest_1, f_rest_2, f_rest_3, f_rest_4, f_rest_5, f_rest_6, f_rest_7, f_rest_8, opacity;
        in.read(reinterpret_cast<char *>(&x), sizeof(float));
        in.read(reinterpret_cast<char *>(&y), sizeof(float));
        in.read(reinterpret_cast<char *>(&z), sizeof(float));
        in.read(reinterpret_cast<char *>(&nx), sizeof(float));
        in.read(reinterpret_cast<char *>(&ny), sizeof(float));
        in.read(reinterpret_cast<char *>(&nz), sizeof(float));
        in.read;}
    */

   /*
    InputGS gs;
    int numPoints = 10;
    gs.means = torch::zeros({numPoints, 3});
    gs.scales = torch::zeros({numPoints, 3});
    gs.quats = torch::zeros({numPoints, 4});
    gs.opacities = torch::logit(0.1f * torch::ones({numPoints, 1}));

    int dimSh = numShbases(inputConfig.shDegree); // 阶数
    torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(inputConfig.device));
        // 输入点的颜色填充至shs张量的第一维，其他维度填充0，分别记录，设置为可求导
    gs.featuresDc = shs.index({Slice(), 0, Slice()}).to(inputConfig.device).requires_grad_(); 
    gs.featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(inputConfig.device).requires_grad_();
    */
    return gs;
}

Config inputConfigFromX(const std::string &inputConfig){
    Config cfg;
    fs::path config(inputConfig);
    if (!fs::exists(config)) throw std::runtime_error("Invalid input config file");
    std::ifstream in(config.string());
    nlohmann::json jsonData;
    try {
        in >> jsonData; 
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse JSON. " << e.what() << std::endl;
        return cfg;
    }
    cfg.numDownscales = jsonData.at("numDownscales");
    cfg.resolutionSchedule = jsonData.at("resolutionSchedule");
    cfg.shDegree = jsonData.at("shDegree");
    cfg.shDegreeInterval = jsonData.at("shDegreeInterval");
    cfg.refineEvery = jsonData.at("refineEvery");
    cfg.warmupLength = jsonData.at("warmupLength");
    cfg.resetAlphaEvery = jsonData.at("resetAlphaEvery");
    cfg.stopSplitAt = int(jsonData.at("maxSteps"))/2;
    cfg.densifyGradThresh = jsonData.at("densifyGradThresh");
    cfg.densifySizeThresh = jsonData.at("densifySizeThresh");
    cfg.stopScreenSizeAt = jsonData.at("stopScreenSizeAt");
    cfg.splitScreenSize = jsonData.at("splitScreenSize");
    cfg.maxSteps = jsonData.at("maxSteps");
    cfg.keepCrs = jsonData.at("keepCrs");
    if (jsonData.at("device") == "cuda"){
        cfg.device = torch::kCUDA;
    }else if (jsonData.at("device") == "cpu"){
        cfg.device = torch::kCPU;
    }else{
        cfg.device = torch::kCUDA;
    }
    return cfg;
    
}