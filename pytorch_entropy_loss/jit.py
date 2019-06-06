from torch.utils.cpp_extension import load
entropy_loss_cuda = load(
    'entropy_loss_cuda', ['entropy_loss_cuda.cpp', 'entropy_loss_cuda_kernel.cu'], verbose=True)
help(entropy_loss_cuda)
