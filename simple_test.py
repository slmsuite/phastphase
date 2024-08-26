from tabnanny import verbose
import torch
from torch.fft import fftn, ifftn, fft2,ifft2
import numpy as np

from phastphase import retrieve


if torch.cuda.is_available():
    tensor_device = torch.device('cuda')
p = 6 #power of 2 for the near field image
overs = 2   #oversampling
N = 2**p    #number of pixels in 1 dimension
x = torch.randn((N,N),dtype=torch.cdouble, device = tensor_device)

d = 2
x[d,d] =  2*N
y = torch.square(torch.abs(fftn(x,(overs*N+1, overs*N+1), norm = 'ortho')))
x_out = retrieve(y, [N,N],grad_tolerance = 1e-9)

print(torch.linalg.vector_norm(x_out - x)/torch.linalg.norm(x))