#include "calcHist.h"

void calcHist(const cv::Mat& grayscaleImg, std::vector<int>& histogram) {
    // Inicializad o histograma com zeros e 256 bins
    histogram.assign(256, 0);

    // Computa o histograma
    for (int row = 0; row < grayscaleImg.rows; ++row) {
        for (int col = 0; col < grayscaleImg.cols; ++col) {
            int pixelValue = grayscaleImg.at<uchar>(row, col);
            histogram[pixelValue]++;
        }
    }
}