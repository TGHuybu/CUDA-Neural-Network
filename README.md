# **CUDA-Neural-Network**

Classification neural network implemented in `C++` and `CUDA`.

## **Worspace**

The project is organized following this structure:
```
repo/
├── include/
│   └── *.h
├── src/
│   ├── *.cpp   -> Base C++ functions
│   └── *.cu    -> CUDA related functions
├── bin/
│   └── nn_main -> Compiled executable file
├── Makefile
└── main.ipynb  -> Google Colab Notebook
```

## **Usage**

To compile the program, navigate to the main repo's folder and run `make`.

To run the program:
```
./nn_main <#-neurons> <#-epochs> <learning-rate> <mode>
# NOTE: Set `mode` to `0` to not use optimized GPU.
```

The Notebook can be used to both compile run the program in Google Colab.

When invoked, the program will run 2 identical models, 1 on CPU, and 1 on GPU (either using the baseline or combined optimized version). 

The program will run the following operations:
- Initial forward calculation:
    - Measure **CPU + GPU** runtime for each layer. 
    - Calculate mean error between 2 versions.
- Train (during each epoch):
    - Measure **CPU + GPU** forward runtime for each layer.
    - Measure **GPU** backward runtime for each layer.
    - Compute cross-entropy loss for each epoch.
- Test (after training):
    - Compute final (classification layer) prediction for **CPU + GPU** model.
    - Calculate mean error between 2 predictions.
