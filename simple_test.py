import torch
from torch.fft import fftn, ifftn, fft2,ifft2
import numpy as np

from phasefast import retrieve


if torch.cuda.is_available():
    tensor_device = torch.device('cuda')
p = 6 #power of 2 for the near field image
overs = 4   #oversampling
N = 2**p    #number of pixels in 1 dimension
x = torch.randn((N,N),dtype=torch.cdouble, device = tensor_device)

d = 1
x[d,d] =  .8*N
x_else = x.clone().detach()
x_else[d,d] = 0
print(f'Proportion of first element to 1-norm of rest of object: {x[d,d]/torch.linalg.vector_norm(x_else, ord = 2)}')
y = torch.square(torch.abs(fftn(x,(overs*N+1, overs*N+1), norm = 'ortho')))
x_in = x
x_mags=torch.abs(x).clone()
x_out = retrieve(y, [N,N])

print(torch.linalg.norm(x_out - x_in)/torch.linalg.norm(x_in))