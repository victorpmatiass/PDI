#include "laplacian.h"


cv::Mat applyLaplacian(const cv::Mat& image) {
    // Define o kernel
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
                       0,  1, 0,
                       1, -4, 1,
                       0,  1, 0);
    // Inicializa com zeros
    cv::Mat laplacian = cv::Mat::zeros(image.size(), CV_32F);

    int rows = image.rows;
    int cols = image.cols;

    for(int i = 1; i < rows - 1; ++i) {
        for(int j = 1; j < cols - 1; ++j) {
            // Extrai a região de interesse 
            cv::Mat roi = image(cv::Range(i-1, i+2), cv::Range(j-1, j+2));
            cv::Mat roiFloat;
            roi.convertTo(roiFloat, CV_32F);
            // Multiplica pelo kernel
            float value = 0.0f;
            for(int m = 0; m < 3; ++m) {
                for(int n = 0; n < 3; ++n) {
                    value += roiFloat.at<float>(m, n) * kernel.at<float>(m, n);
                }
            }
            laplacian.at<float>(i, j) = value;
        }
    }

    return laplacian;
}
