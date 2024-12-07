# Variáveis
CXX = g++
CXXFLAGS = `pkg-config --cflags --libs opencv4`

# Alvo padrão
# cpp:
# 	$(CXX) -o $(file) $(file).cpp $(CXXFLAGS)

# Variáveis
#CXX = g++
# CXXFLAGS = `pkg-config --cflags --libs opencv4`

# Alvo padrão
cpp:
	$(CXX) -o $(word 2, $(MAKECMDGOALS)) $(word 2, $(MAKECMDGOALS)).cpp $(CXXFLAGS)
