#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat redimensionarBilinear(const cv::Mat& imagem, int nova_largura, int nova_altura) {
    cv::Mat imagem_redimensionada(nova_altura, nova_largura, imagem.type());

    float escala_x = static_cast<float>(imagem.cols) / nova_largura;
    float escala_y = static_cast<float>(imagem.rows) / nova_altura;

    for (int y = 0; y < nova_altura; ++y) {
        for (int x = 0; x < nova_largura; ++x) {
            // Coordenadas no espaço original
            float origem_x = x * escala_x;
            float origem_y = y * escala_y;

            // Pega os índices dos 4 pixels vizinhos
            int x0 = static_cast<int>(origem_x);
            int x1 = std::min(x0 + 1, imagem.cols - 1);
            int y0 = static_cast<int>(origem_y);
            int y1 = std::min(y0 + 1, imagem.rows - 1);

            // Calcula os pesos para interpolação
            float wx = origem_x - x0;
            float wy = origem_y - y0;

            // Interpolação bilinear
            for (int c = 0; c < imagem.channels(); ++c) {
                float valor =
                    (1 - wx) * (1 - wy) * imagem.at<cv::Vec3b>(y0, x0)[c] +
                    wx * (1 - wy) * imagem.at<cv::Vec3b>(y0, x1)[c] +
                    (1 - wx) * wy * imagem.at<cv::Vec3b>(y1, x0)[c] +
                    wx * wy * imagem.at<cv::Vec3b>(y1, x1)[c];
                imagem_redimensionada.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(valor);
            }
        }
    }

    return imagem_redimensionada;
}

int main(void) {
    int dpi_original, dpi_desejado;
    std::string img_path;

    std::cout << "Digite o caminho da imagem: ";
    std::cin >> img_path;
    
    cv::Mat imagem = cv::imread(img_path);
    if (imagem.empty()) {
        std::cerr << "Erro ao carregar a imagem!" << std::endl;
        return -1;
    }

    std::cout << "Digite o DPI original da imagem: ";
    std::cin >> dpi_original;

    std::cout << "Digite o DPI desejado: ";
    std::cin >> dpi_desejado;

    float fator_escala = static_cast<float>(dpi_desejado) / dpi_original;
    int nova_largura = static_cast<int>(imagem.cols * fator_escala);
    int nova_altura = static_cast<int>(imagem.rows * fator_escala);

    // Redimensiona a imagem manualmente
    cv::Mat imagem_redimensionada = redimensionarBilinear(imagem, nova_largura, nova_altura);

    // Exibe as imagens
    cv::namedWindow("Imagem original", cv::WINDOW_NORMAL);
    cv::resizeWindow("Imagem original", 800, 600);
    cv::imshow("Imagem original", imagem);

    cv::namedWindow("Imagem Redimensionada", cv::WINDOW_NORMAL);
    cv::resizeWindow("Imagem Redimensionada", 800, 600);
    cv::imshow("Imagem Redimensionada", imagem_redimensionada);

    cv::waitKey(0);

    std::cout << "Dimensão imagem original: " 
              << imagem.size().width << "x" 
              << imagem.size().height << std::endl;

    std::cout << "Imagem redimensionada para: " 
              << imagem_redimensionada.size().width << "x" 
              << imagem_redimensionada.size().height << std::endl;

    return 0;
}
