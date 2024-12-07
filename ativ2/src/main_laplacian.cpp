#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "laplacian.h"

int main() {
    // Paths to input images
    std::string luaPath = "../images/lua.tif";
    std::string dolarPath = "../images/dollar.tif";
    std::string graosPath = "../images/graos.tif";

    // Load images in grayscale
    cv::Mat lua = cv::imread(luaPath, cv::IMREAD_GRAYSCALE);
    cv::Mat dolar = cv::imread(dolarPath, cv::IMREAD_GRAYSCALE);
    cv::Mat graos = cv::imread(graosPath, cv::IMREAD_GRAYSCALE);

    // List of images to process
    std::vector<std::pair<std::string, cv::Mat>> images = {
        {"lua", lua},
        {"graos", graos},
        {"dolar", dolar}
    };

    for(auto &imgPair : images) {
        std::string imgName = imgPair.first;
        cv::Mat img = imgPair.second;

        // Apply Laplacian
        cv::Mat laplacian = applyLaplacian(img);

        // Adjust Laplacian
        cv::Mat laplacianAdjusted;
        double minVal, maxVal;
        cv::minMaxLoc(laplacian, &minVal, &maxVal);
        laplacianAdjusted = (laplacian - minVal) / (maxVal - minVal) * 255.0;
        laplacianAdjusted.convertTo(laplacianAdjusted, CV_8U);

        // Realce Image
        double c = -1.0;
        cv::Mat imagemRealcada;
        cv::Mat laplacianScaled;
        laplacian.convertTo(laplacianScaled, CV_8U);
        cv::addWeighted(img, 1.0, laplacianScaled, c, 0.0, imagemRealcada);

        cv::Mat display;
        cv::hconcat(std::vector<cv::Mat>{
            img, 
            imagemRealcada, 
            cv::Mat::zeros(img.size(), img.type()), 
            laplacianAdjusted
        }, display);


        int width = img.cols;
        int height = img.rows;

        // Clone individual images to add titles
        cv::Mat imgOriginalDisplay, imgRealcadaDisplay, imgLaplacianDisplay, imgAdjustedDisplay;
        cv::cvtColor(img, imgOriginalDisplay, cv::COLOR_GRAY2BGR);
        cv::cvtColor(imagemRealcada, imgRealcadaDisplay, cv::COLOR_GRAY2BGR);
        cv::cvtColor(laplacianScaled, imgLaplacianDisplay, cv::COLOR_GRAY2BGR);
        cv::cvtColor(laplacianAdjusted, imgAdjustedDisplay, cv::COLOR_GRAY2BGR);

        // Add titles
        cv::putText(imgOriginalDisplay, "Imagem Original", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(imgRealcadaDisplay, "Imagem Realcada", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(imgLaplacianDisplay, "Laplaciano sem ajuste", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(imgAdjustedDisplay, "Laplaciano com ajuste", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);

        // Concatenate images horizontally
        cv::Mat concatenated;
        cv::hconcat(std::vector<cv::Mat>{
            imgOriginalDisplay, 
            imgRealcadaDisplay, 
            imgLaplacianDisplay, 
            imgAdjustedDisplay
        }, concatenated);

        // Display the concatenated image
        std::string windowName = imgName + " - Processing Results";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, concatenated);

        // Plot Histograms using OpenCV
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = { range };
        bool uniform = true, accumulate = false;

        cv::Mat histOriginal, histRealcada;

        // Compute histograms
        cv::calcHist(&img, 1, 0, cv::Mat(), histOriginal, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&imagemRealcada, 1, 0, cv::Mat(), histRealcada, 1, &histSize, &histRange, uniform, accumulate);

        // Create an image to display histograms
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound((double) hist_w / histSize);

        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255,255,255));

        // Normalize the histograms to fit the histImage height
        cv::normalize(histOriginal, histOriginal, 0, histImage.rows, cv::NORM_MINMAX);
        cv::normalize(histRealcada, histRealcada, 0, histImage.rows, cv::NORM_MINMAX);

        // Draw the histograms
        for(int i = 1; i < histSize; ++i) {
            cv::line(histImage, 
                     cv::Point(bin_w*(i-1), hist_h - cvRound(histOriginal.at<float>(i-1))),
                     cv::Point(bin_w*(i), hist_h - cvRound(histOriginal.at<float>(i))),
                     cv::Scalar(0,0,0), 2);
            cv::line(histImage, 
                     cv::Point(bin_w*(i-1), hist_h - cvRound(histRealcada.at<float>(i-1))),
                     cv::Point(bin_w*(i), hist_h - cvRound(histRealcada.at<float>(i))),
                     cv::Scalar(255,0,0), 2);
        }

        // Add titles and legends
        cv::putText(histImage, "Histograma da Imagem Original", cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
        cv::putText(histImage, "Histograma da Imagem Realcada", cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);

        // Display the histogram
        std::string histWindow = imgName + " - Histograms";
        cv::namedWindow(histWindow, cv::WINDOW_NORMAL);
        cv::imshow(histWindow, histImage);
    }

    // Wait for key press
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
