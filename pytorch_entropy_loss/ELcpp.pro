TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    entropy_loss_cuda.cpp \
    entropy_loss_cuda_kernel.cu

INCLUDEPATH += "/home/nvidia/anaconda3/envs/pytorch-env/lib/python3.6/site-packages/torch/include/"
INCLUDEPATH += "/home/nvidia/anaconda3/envs/pytorch-env/lib/python3.6/site-packages/torch/include/torch/csrc/api/include/"
INCLUDEPATH += "/usr/local/cuda/include/"
