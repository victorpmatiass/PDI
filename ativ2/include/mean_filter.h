#ifndef MEAN_FILTER_H
#define MEAN_FILTER_H

#include <opencv2/opencv.hpp>

cv::Mat applyMeanFilter(const cv::Mat &image, int m);

#endif 
