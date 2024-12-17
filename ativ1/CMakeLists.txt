# Especifica a versão mínima do CMake requerida
cmake_minimum_required(VERSION 3.10)

# Nome do projeto e linguagem utilizada
project(projeto-1 LANGUAGES CXX)

# Define o padrão C++ a ser utilizado (C++11, C++17, etc.)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Encontra o pacote OpenCV
find_package(OpenCV REQUIRED)

# Inclui os diretórios de cabeçalhos do OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Adiciona o executável a ser gerado
add_executable(projeto-1 projeto-1.cpp)

# Liga o executável com as bibliotecas do OpenCV
target_link_libraries(projeto-1 ${OpenCV_LIBS})

