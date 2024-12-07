// src/mean_filter.cpp
#include "mean_filter.h"

// Implementação da função applyMeanFilter
cv::Mat applyMeanFilter(const cv::Mat &image, int m) {
    if (m % 2 == 0) m += 1; 
    cv::Mat filtrada;
    cv::blur(image, filtrada, cv::Size(m, m));
    return filtrada;
}
