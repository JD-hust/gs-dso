#ifndef CV_UTILS
#define CV_UTILS

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

cv::Mat imreadRGB(const std::string &filename);
void imwriteRGB(const std::string &filename, const cv::Mat &image);
cv::Mat floatNxNtensorToMat(const torch::Tensor &t);
torch::Tensor floatNxNMatToTensor(const cv::Mat &m);
cv::Mat tensorToImage(const torch::Tensor &t);
cv::Mat tensorTodepthImage(const torch::Tensor &t);
cv::Mat tensorToDepthImage(const torch::Tensor &t);
cv::Mat depthTensorToImage(const torch::Tensor &t, int id);
torch::Tensor imageToTensor(const cv::Mat &image);

#endif