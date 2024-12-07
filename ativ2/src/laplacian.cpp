#include "laplacian.h"

// Function to apply Laplacian filter manually
cv::Mat applyLaplacian(const cv::Mat& image) {
    // Define the Laplacian kernel
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
                       0,  1, 0,
                       1, -4, 1,
                       0,  1, 0);
    // Initialize the output Laplacian image with zeros
    cv::Mat laplacian = cv::Mat::zeros(image.size(), CV_32F);

    int rows = image.rows;
    int cols = image.cols;

    for(int i = 1; i < rows - 1; ++i) {
        for(int j = 1; j < cols - 1; ++j) {
            // Extract the Region of Interest (3x3)
            cv::Mat roi = image(cv::Range(i-1, i+2), cv::Range(j-1, j+2));
            // Convert ROI to float for multiplication
            cv::Mat roiFloat;
            roi.convertTo(roiFloat, CV_32F);
            // Multiply ROI with kernel
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
