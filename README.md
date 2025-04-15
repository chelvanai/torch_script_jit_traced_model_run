## Libtorch (torch C++ works)

**use pytorch models without any dependence in local machine.**

It is developed in Windows 11 machine, by **WSL (windows subsystem linux)**. so it gives executable output for linux platform only.

## Setting Up WSL for CUDA Development

This guide will help you set up a development environment on your Windows machine using WSL with Ubuntu for CUDA programming.


1. **Prepare WSL on Windows**
   - Ensure you have WSL installed on your Windows machine. Follow online instructions if needed.


2. **Install Ubuntu on WSL**
   - Download and install Ubuntu from the Microsoft Store or through WSL.


3. **Install Required Packages in Ubuntu**
   - Open WSL and install the following packages:

     ```bash
     sudo apt-get update
     sudo apt install gcc
     sudo apt install nvidia-cuda-toolkit
     sudo apt-get install g++
     sudo apt-get install cmake
     sudo apt install build-essential -y
     ```


4. **Install Specific Version of Tools**
   - Ensure the following versions are installed:
      - **gcc:** 13.3.0
      - **g++:** 13.3.0
      - **cmake:** 3.28.3
      - **nvcc:** Cuda compilation tools, release 12.0, V12.0.140


5. **Install NVIDIA CUDA Toolkit on Windows**
   - Download and install CUDA Toolkit version 12.04 for Windows to enable graphics card access from WSL.


6. **Download Libtorch**
   - Download Libtorch version 2.6.0+cu124 for Linux.
   - Extract it and place it in your project folder.


7. **Ue torch script to convert the pytorch model to jit traced version**
    - First you should convert the pytorch model to jit traced version.
    - Run the `pytorch_model_jit_trace_convert.py` file
   
TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.
We provide tools to incrementally transition a model from a pure Python program to a TorchScript program that can be run independently from Python, such as in a standalone C++ program.


8. **Build and Run Your Project**
   - Open CMD and start WSL.
   - Navigate to your project folder:
     ```bash
     cd /path/to/your/project
     ```
   - Follow these steps to build and run your project:

     ```bash
     mkdir build
     cd build
     cmake ..
     make
     ./resnet_run
     ```
     
**According to this works we could compile cuda based torch c++ code in linux sub system**

## Issues when try Libtorch on Windows without WSL

When attempting to use Libtorch on a Windows environment, the following issues may arise:

1. **Missing NVTX Files with CUDA 12.4**
   - When using CUDA version 12.4, you may encounter errors due to missing NVTX files that are needed for certain operations.

2. **Compatibility Issues with CUDA 11.8 and MSVC 2022**
   - Although CUDA version 11.8 contains the necessary NVTX files, it may not compile correctly with the latest Microsoft MSVC C Compiler 2022 due to version mismatches.

3. **Compiler Limitations on Windows**
   - On Windows, only the MSVC compiler is supported for certain operations; using the GCC compiler is not an option.

These issues may affect the compatibility and build process when working with Libtorch in a Windows environment. 
