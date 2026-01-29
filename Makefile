CXX := g++
CXX_FLAGS := -std=c++17 -ggdb
# NVCC := nvcc

BIN := bin
SRC := src
INCLUDE := include 

EXECUTABLE := nn_main

all: $(BIN)/$(EXECUTABLE)

run: clean all
	clear 
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) $^ -o $@

clean: 
	-rm $(BIN)/*
