#include "mean_filter.h"

cv::Mat applyMeanFilter(const cv::Mat& img, int m) {
    // Ensure kernel size is odd
    if (m % 2 == 0) {
        m += 1;
    }

    int height = img.rows;
    int width = img.cols;

    cv::Mat kernel = cv::Mat::ones(m, m, CV_32F) / (float)(m * m);

    int pad = m / 2;
    cv::Mat paddedImg;
    cv::copyMakeBorder(img, paddedImg, pad, pad, pad, pad, cv::BORDER_CONSTANT, 0);
    cv::Mat filteredImg = cv::Mat::zeros(height, width, CV_8UC1);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Extract the (m x m) window from the padded image
            cv::Mat window = paddedImg(cv::Rect(j, i, m, m));

            // Perform element-wise multiplication with the kernel and sum the results
            cv::Mat result;
            cv::multiply(window, kernel, result, 1, CV_32F);

            float meanValue = cv::sum(result)[0];
            filteredImg.at<uchar>(i, j) = cv::saturate_cast<uchar>(meanValue);
        }
    }

    return filteredImg;
}