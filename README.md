# Fast-SCNN-libtorch
Fast-SCNN inference code with libtorch
# Environment
- Ubuntu 18.04
- Cuda 10.2, V10.2.89
- libtorch 1.10.2

# How to run
1. change "Torch_DIR" variable in Cmakelist
2. mkdir build && cd build 
3. cmake .. make
4. in terminal run ‘./inference <path-to-exported-script-module> <path-to-img>’
