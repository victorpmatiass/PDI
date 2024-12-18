
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "mean_filter.h"

using namespace std;
using namespace cv;

struct ImageSet {
    Mat original_labeled;        
    Mat filtradas_labeled;      
    Mat enhancement_labeled;     
};

// Adiciona os rótulos
Mat add_label(const Mat &img, const string &label) {
    Mat img_labeled;
    cv::cvtColor(img, img_labeled, COLOR_GRAY2BGR);
    
    // Parâmetros 
    int font = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 1;
    int baseline = 0;
    // posicão e tamanho do texto
    Size textSize = getTextSize(label, font, fontScale, thickness, &baseline);
    Point textOrg(10, textSize.height + 10);
    rectangle(img_labeled, textOrg + Point(0, baseline), 
              textOrg + Point(textSize.width, -textSize.height), 
              Scalar(0,0,0), FILLED);
    
    putText(img_labeled, label, textOrg, font, fontScale, Scalar(255,255,255), thickness, LINE_AA);
    
    return img_labeled;
}

int main() {
    // vetor com pares de caminho e rótulo das imagens
    vector<pair<string, string>> imagens = {
        {"../images/a.tif", "Imagem A"},
        {"../images/lua.tif", "Imagem Lua"},
        {"../images/quadrado.tif", "Imagem Quadrado"}
    };

    // Definindo parâmetros do filtro que serão usado
    vector<int> filter_sizes = {3, 5, 9, 15, 35}; 
    // Determinar o tamanho comum (usar o menor tamanho entre as imagens para evitar distorção)
    int common_width = INT32_MAX;
    int common_height = INT32_MAX;
    // Carregar todas as imagens primeiro para determinar o tamanho comum
    vector<Mat> loaded_images;
    for(auto &p : imagens){
        Mat img = imread(p.first, IMREAD_GRAYSCALE);
        if(img.empty()){
            cout << "Não foi possível carregar a imagem: " << p.first << endl;
            return -1;
        }
        loaded_images.push_back(img);
        common_width = min(common_width, img.cols);
        common_height = min(common_height, img.rows);
    }

    Size common_size(common_width, common_height);

    // Vetor para armazenar os conjuntos de imagens processadas
    vector<ImageSet> imagens_processadas;

    // Processar cada imagem
    for(size_t i = 0; i < loaded_images.size(); i++){
        Mat img = loaded_images[i];
        string imageName = imagens[i].second;

        // Redimensionar para tamanho comum
        resize(img, img, common_size);

        // Garantir que todas as imagens têm o mesmo tipo
        img.convertTo(img, CV_8U);

        // Aplicar filtros de média com diferentes tamanhos
        vector<Mat> filtradas_labeled;
        for(auto &m : filter_sizes){
            Mat filtrada = applyMeanFilter(img, m); // aplicar filtro da média
            string label = "Filtrada (m = " + to_string(m) + ")";
            Mat filtrada_labeled = add_label(filtrada, label);
            filtradas_labeled.push_back(filtrada_labeled);
        }

        // Filtrada com m = 3
        Mat filtrada_m = applyMeanFilter(img, 3);
        Mat filtrada_m_labeled = add_label(filtrada_m, "Filtrada (m = 9)");

        // Calcular máscara = original - filtrada_m
        Mat mascara;
        {
            Mat img16, filt16;
            img.convertTo(img16, CV_16S);       
            filtrada_m.convertTo(filt16, CV_16S);
            mascara = img16 - filt16;   
        }

        // Realça imagem: original + k*mascara
        int k = 1; 
        Mat realcada;
        {
            Mat img16, realcada_temp;
            img.convertTo(img16, CV_16S);
            realcada_temp = img16 + k * mascara;
            realcada_temp.convertTo(realcada, CV_8U);
            mascara.convertTo(mascara, CV_8U);
        }

        // Adicionar rótulos
        Mat mascara_labeled = add_label(mascara, "Máscara");
        Mat realcada_labeled = add_label(realcada, "Realçada");

        // --- Organizar as Imagens ---
        Mat row1, row2, row3, grid;
        hconcat(vector<Mat>{add_label(img, "Original"), filtradas_labeled[0]}, row1);
        hconcat(vector<Mat>{filtradas_labeled[1], filtradas_labeled[2]}, row2);
        hconcat(vector<Mat>{filtradas_labeled[3], filtradas_labeled[4]}, row3);
        vconcat(vector<Mat>{row1, row2, row3}, grid);

        Mat enhancement_grid;
        hconcat(vector<Mat>{mascara_labeled, realcada_labeled}, enhancement_grid);

        ImageSet set;
        set.original_labeled = add_label(img, "Original");
        set.filtradas_labeled = grid; // 3x2 grid com filtragens
        set.enhancement_labeled = enhancement_grid; // 1x2 grid com máscara e realçada

        imagens_processadas.push_back(set);

        // Adiciona rótulos
        string title_filters = "Comparacao das Filtragens - " + imageName;
        int font = FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        int thickness = 2;
        Size textSize = getTextSize(title_filters, font, fontScale, thickness, 0);
        Mat titleImg_filters(textSize.height + 20, grid.cols, grid.type(), Scalar(255,255,255)); 
        putText(titleImg_filters, title_filters, Point((grid.cols - textSize.width)/2, textSize.height + 10), 
                font, fontScale, Scalar(0,0,0), thickness, LINE_AA);

        // Empilhar verticalmente o título e a grade de filtragens
        Mat final_image_filters;
        vconcat(vector<Mat>{titleImg_filters, grid}, final_image_filters);

        // Exibe a comparação das filtragens
        namedWindow(title_filters, WINDOW_NORMAL);
        imshow(title_filters, final_image_filters);

        // Adiciona o rótulo para o realce da imagem
        string title_enhancement = "Realce da Imagem - " + imageName;
        Size textSize_enhancement = getTextSize(title_enhancement, font, fontScale, thickness, 0);
        Mat titleImg_enhancement(textSize_enhancement.height + 20, enhancement_grid.cols, enhancement_grid.type(), Scalar(255,255,255)); // fundo branco para o título
        putText(titleImg_enhancement, title_enhancement, Point((enhancement_grid.cols - textSize_enhancement.width)/2, textSize_enhancement.height + 10), 
                font, fontScale, Scalar(0,0,0), thickness, LINE_AA);

        // Empilhar verticalmente
        Mat final_image_enhancement;
        vconcat(vector<Mat>{titleImg_enhancement, enhancement_grid}, final_image_enhancement);

        // realce da imagem
        namedWindow(title_enhancement, WINDOW_NORMAL);
        imshow(title_enhancement, final_image_enhancement);
    }
    waitKey(0);

    return 0;
}
