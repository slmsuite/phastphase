import torch
from torch.fft import fftn
import numpy as np
from phastphase import retrieve


if torch.cuda.is_available():
    tensor_device = torch.device('cpu')
else:
    tensor_device = torch.device('cpu')

p = 8                       #power of 2 for the near field image
overs = 2                   #oversampling
N = 2**p                    #number of pixels in 1 dimension
x = torch.randn((N,N),
                dtype=torch.cdouble, 
                device = tensor_device
                )
mask = np.ones((N,N))
d = 1
x[d,d] =  2*N
y = torch.square(torch.abs(fftn(x,(overs*N, overs*N), norm = 'ortho')))
x_out = retrieve(y, [N,N],grad_tolerance = 1e-12,verbose=True,tr_max_iter = 100, support_mask= mask)
#print(torch.linalg.vector_norm(x_out - x)/torch.linalg.norm(x))