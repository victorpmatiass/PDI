#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "laplacian.h"
#include "calcHist.h"

int main() {
    // Leitura das imagens
    std::string luaPath = "../images/lua.tif";
    std::string dolarPath = "../images/dollar.tif";
    std::string graosPath = "../images/graos.tif";
    cv::Mat lua = cv::imread(luaPath, cv::IMREAD_GRAYSCALE);
    cv::Mat dolar = cv::imread(dolarPath, cv::IMREAD_GRAYSCALE);
    cv::Mat graos = cv::imread(graosPath, cv::IMREAD_GRAYSCALE);

    // Imagens para processamento 
    std::vector<std::pair<std::string, cv::Mat>> images = {
        {"lua", lua},
        {"graos", graos},
        {"dolar", dolar}
    };

    for(auto &imgPair : images) {
        std::string imgName = imgPair.first;
        cv::Mat img = imgPair.second;

        // Aplicacão do Laplaciano
        cv::Mat laplacian = applyLaplacian(img);
        cv::Mat laplacianAdjusted;
        double minVal, maxVal;
        cv::minMaxLoc(laplacian, &minVal, &maxVal);
        laplacianAdjusted = (laplacian - minVal) / (maxVal - minVal) * 255.0;
        laplacianAdjusted.convertTo(laplacianAdjusted, CV_8U);

        // Realce da imagem
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

        // Cria as imagens para exibição e adiciona títulos
        cv::Mat imgOriginalDisplay, imgRealcadaDisplay, imgLaplacianDisplay, imgAdjustedDisplay;
        cv::cvtColor(img, imgOriginalDisplay, cv::COLOR_GRAY2BGR);
        cv::cvtColor(imagemRealcada, imgRealcadaDisplay, cv::COLOR_GRAY2BGR);
        cv::cvtColor(laplacianScaled, imgLaplacianDisplay, cv::COLOR_GRAY2BGR);
        cv::cvtColor(laplacianAdjusted, imgAdjustedDisplay, cv::COLOR_GRAY2BGR);
        cv::putText(imgOriginalDisplay, "Imagem Original", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(imgRealcadaDisplay, "Imagem Realcada", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(imgLaplacianDisplay, "Laplaciano sem ajuste", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(imgAdjustedDisplay, "Laplaciano com ajuste", cv::Point(10, 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);

        // Concatena as imagens horizontalmente e exibe
        cv::Mat concatenated;
        cv::hconcat(std::vector<cv::Mat>{
            imgOriginalDisplay, 
            imgRealcadaDisplay, 
            imgLaplacianDisplay, 
            imgAdjustedDisplay
        }, concatenated);
        std::string windowName = imgName + " - Processing Results";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, concatenated);

        std::vector<int> histOriginal;
        std::vector<int> histRealcada;
        calcHist(img, histOriginal);
        calcHist(imagemRealcada, histRealcada);

        int histSize = 256;

        // Exibe os histogramas
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound((double) hist_w / histSize);

        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255,255,255));

        // Normaliza os histogramas
        cv::normalize(histOriginal, histOriginal, 0, histImage.rows, cv::NORM_MINMAX);
        cv::normalize(histRealcada, histRealcada, 0, histImage.rows, cv::NORM_MINMAX);

        // Gera os histogramas e adiciona títulos e legendas par exibicão
        for(int i = 0; i < histSize; ++i) {
            cv::line(histImage, 
                     cv::Point(bin_w * i, hist_h),
                     cv::Point(bin_w * i, hist_h - histOriginal[i]),
                     cv::Scalar(255,0,0), 2);
            cv::line(histImage, 
                     cv::Point(bin_w * i, hist_h),
                     cv::Point(bin_w * i, hist_h - histRealcada[i]),
                     cv::Scalar(0,255,0), 2);
        }
        cv::putText(histImage, "Histograma da Imagem Original", cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);
        cv::putText(histImage, "Histograma da Imagem Realcada", cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        std::string histWindow = imgName + " - Histograms";
        cv::namedWindow(histWindow, cv::WINDOW_NORMAL);
        cv::imshow(histWindow, histImage);
    }
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
