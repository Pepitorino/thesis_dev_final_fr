#pragma once
#include <vector>
#include <opencv2/core.hpp>

extern "C"
void project_pixels_cuda(
    int width,
    int height,
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d,
    const std::vector<double>& e,
    const std::vector<double>& f,
    const std::vector<int>& types,
    const std::vector<double>& weights,
    cv::Mat& output_image,
    double& frontier_score,
    double& roi_score,
    double& occupied_score
);
