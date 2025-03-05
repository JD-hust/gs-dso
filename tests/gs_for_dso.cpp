#include <iostream>
#include <queue>
#include <vector>
#include <thread>
#include "input_data.hpp"
#include <filesystem>
#include <cxxopts.hpp>
#include "gsmap.hpp"
#include "cv_utils.hpp"
#include <sstream>
#include <string>
// 测试代码
Camera cameraFromPose(std::string pose1){
    Camera cam;
    std::vector<int> numbers;

    std::istringstream stream(pose1);
    std::string temp;

    while (stream >> temp) {
        numbers.push_back(std::stoi(temp));
    }

    torch::Tensor pose = torch::from_blob(numbers.data(), {static_cast<long>(numbers.size())}, torch::kInt32);

    // 提取平移和四元数
    float tx = pose[0].item<int>();
    float ty = pose[1].item<int>();
    float tz = pose[2].item<int>();
    float qx = pose[3].item<int>();
    float qy = pose[4].item<int>();
    float qz = pose[5].item<int>();
    float qw = pose[6].item<int>();

    // 计算旋转矩阵
    torch::Tensor rotation = torch::zeros({3, 3}, torch::kFloat32);
    rotation[0][0] = 1 - 2 * (qy * qy + qz * qz);
    rotation[0][1] = 2 * (qx * qy - qz * qw);
    rotation[0][2] = 2 * (qx * qz + qy * qw);
    rotation[1][0] = 2 * (qx * qy + qz * qw);
    rotation[1][1] = 1 - 2 * (qx * qx + qz * qz);
    rotation[1][2] = 2 * (qy * qz - qx * qw);
    rotation[2][0] = 2 * (qx * qz - qy * qw);
    rotation[2][1] = 2 * (qy * qz + qx * qw);
    rotation[2][2] = 1 - 2 * (qx * qx + qy * qy);

    // 创建 4x4 的 camToWorld 矩阵
    torch::Tensor camToWorld = torch::eye(4, torch::kFloat32);
    camToWorld.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, rotation);
    camToWorld[0][3] = tx;
    camToWorld[1][3] = ty;
    camToWorld[2][3] = tz; 
    cam.id = -1;
    cam.width = 640;
    cam.height = 480;
    cam.fx = 500;
    cam.fy = 500;
    cam.cx = 320;
    cam.cy = 240;
    cam.filePath = "pose";
    cam.camToWorld = camToWorld;
    return cam;
}
int main(){
    std::string inputply = "";
    std::string inputconfig = "";
    GSMap gsmap(inputply,inputconfig);
    std::thread t1(&GSMap::run, &gsmap);
    
    while(1){
        std::cout << "Enter pose: ";
        std::string poseInput;
        std::getline(std::cin, poseInput);
        Camera cam = cameraFromPose(poseInput);

        std::lock_guard<std::mutex> lock(gsmap.camQueueMutex);
        gsmap.cameraQueue.push(cam);
        gsmap.conditionVar.notify_one(); 
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (gsmap.depthQueue.empty());
        std::cout << "depthQueue size: " << gsmap.depthQueue.size() << std::endl;
        if (!gsmap.cameraQueue.empty()){
            torch::Tensor depth = gsmap.depthQueue.front();
            gsmap.depthQueue.pop();
            // // 显示深度图
            // cv::Mat depthMat = tensorTodepthImage(depth.squeeze().to(torch::kCPU));
            // cv::imshow("depth", depthMat);
            // cv::waitKey(0);
        }
    }
}
