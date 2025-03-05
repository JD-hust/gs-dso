#include "cv_utils.hpp"
#include <filesystem>

namespace fs = std::filesystem;

cv::Mat imreadRGB(const std::string &filename){
    cv::Mat cImg = cv::imread(filename);
    cv::cvtColor(cImg, cImg, cv::COLOR_BGR2RGB);
    return cImg;
}

void imwriteRGB(const std::string &filename, const cv::Mat &image){
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, rgb);
}

cv::Mat floatNxNtensorToMat(const torch::Tensor &t){
    return cv::Mat(t.size(0), t.size(1), CV_32F, t.data_ptr());
}

torch::Tensor floatNxNMatToTensor(const cv::Mat &m){
    return torch::from_blob(m.data, { m.rows, m.cols }, torch::kFloat32).clone();
}

cv::Mat tensorToImage(const torch::Tensor &t){
    int h = t.sizes()[0];
    int w = t.sizes()[1];
    int c = t.sizes()[2];

    int type = CV_8UC3;
    if (c != 3) throw std::runtime_error("Only images with 3 channels are supported");

    cv::Mat image(h, w, type);
    torch::Tensor scaledTensor = (t * 255.0).toType(torch::kU8);
    uint8_t* dataPtr = static_cast<uint8_t*>(scaledTensor.data_ptr());
    std::copy(dataPtr, dataPtr + (w * h * c), image.data);

    return image;
}

cv::Mat tensorTodepthImage(const torch::Tensor &t) {
    int h = t.sizes()[0];
    int w = t.sizes()[1];
    int c = t.sizes()[2];

    if (c != 1) throw std::runtime_error("Only images with 1 channel are supported");

    // Convert tensor to 16-bit unsigned integer type
    torch::Tensor scaledTensor = t.toType(torch::kUInt16);

    // Create a cv::Mat with the same dimensions and type
    cv::Mat image(h, w, CV_16UC1);

    // Copy data from tensor to cv::Mat
    uint16_t* dataPtr = static_cast<uint16_t*>(scaledTensor.data_ptr());
    std::copy(dataPtr, dataPtr + (w * h * c), reinterpret_cast<uint16_t*>(image.data));

    return image;
}

cv::Mat tensorToDepthImage(const torch::Tensor &t) {
    // debug: 直接显示深度图像:tensor转为cv::Mat，并将深度值转为0-255
    torch::Tensor depth_2 = t.squeeze(); // 去掉最后一个维度，使其变为 h*w
    
    // 获取张量的尺寸
    int height = depth_2.size(0);
    int width = depth_2.size(1);

    // 将张量转换为 OpenCV 的 Mat 格式
    cv::Mat depth_mat(height, width, CV_32F, depth_2.data_ptr<float>());
    cv::Mat depth_mat_16u;
    double min_val, max_val;
    // 获取深度图的最小值和最大值
    max_val = depth_2.max().item<float>();
    min_val = depth_2.min().item<float>();       
    // 归一化深度值到 0-65535
    cv::normalize(depth_mat, depth_mat_16u, 0, 65535, cv::NORM_MINMAX);

    // 尺度因子
    // depth_mat_16u = depth_mat * 6000;

    // 将归一化后的图像转换为 uint16 类型
    depth_mat_16u.convertTo(depth_mat_16u, CV_16U);

    return depth_mat_16u;
}

cv::Mat depthTensorToImage(const torch::Tensor &t, int id){
    int h = t.sizes()[0];
    int w = t.sizes()[1];
    int c = t.sizes()[2];
    
    if (c != 1) throw std::runtime_error("Only tensors with 1 channel are supported!");
    
    int type = CV_8UC1;
    cv::Mat image(h, w, type);
    torch::Tensor scaledTensor = (t * 255 / 8).toType(torch::kU8);
    uint8_t* dataPtr = static_cast<uint8_t*>(scaledTensor.data_ptr());
    std::copy(dataPtr, dataPtr + (w * h * c), image.data);

    cv::Mat colorImage;
    cv::applyColorMap(image, colorImage, cv::COLORMAP_JET);
    
    // static int num = 0;
    // std::string path = "/home/dj/projects/gs-dso-github/gsdso/output/depth/";
    // if (!fs::is_directory(path)){
    //     std::cout << "depth folder not exist, try to create it!" << std::endl;
    //     int flag = fs::create_directory(path);
    //     if(flag == false) std::cout << "Fail to create depth storage directory!" << std::endl;
    //     else std::cout << "Success to create depth storage directory!" << std::endl;
    // }
    // cv::imwrite(path + std::to_string(id) + ".png", colorImage);
    
    return colorImage;
}

torch::Tensor imageToTensor(const cv::Mat &image){
    torch::Tensor img = torch::from_blob(image.data, { image.rows, image.cols, image.dims + 1 }, torch::kU8);
    return (img.toType(torch::kFloat32) / 255.0f);
}

