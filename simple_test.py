import scipy.optimize
import torch
from torch.fft import fftn
import numpy as np
import torchmin.trustregion
from phastphase import retrieve, SOS_loss2, SOS_loss3, masked_loss, refine_
from phastphase.script_modules import create_wind_eval
import torchmin
import scipy
import math
if torch.cuda.is_available():
    tensor_device = torch.device('cpu')
else:
    tensor_device = torch.device('cpu')

p = 6                       #power of 2 for the near field image
overs = 4                   #oversampling
N = 2**p                 #number of pixels in 1 dimension
x = torch.randn((N,N),
                dtype=torch.cdouble, 
                device = tensor_device
                )

mask = np.ones((N,N))
d = 0
x[d,d] =  1.3*N

def poisson_loss_func(x,y,cost_lam,ind1,ind2):
    f = torch.abs(fftn(x, s=y.shape, norm='ortho'))
    return torch.square(torch.linalg.vector_norm(f - y)) + cost_lam*torch.square(torch.imag(x[ind1,ind2]))
y = torch.square(torch.abs(fftn(x,(overs*N, overs*N), norm = 'ortho')))
oversample_r = (2**math.ceil(math.log2(overs*N-5)))/(overs*N-5)
x_out = retrieve(y, [N,N],grad_tolerance = 1e-9,reference_point = [d,d], verbose=True,tr_max_iter = 1000, oversample_ratio=1, adam_iters=0, return_each_iter = True)

#x_out =refine_(x_out, y, (0,0), torch.ones_like(x), x.shape, (0,0), loss_func=poisson_loss_func)
print(torch.linalg.vector_norm(x_out[0] - x)/N**2)

def loss_lam(x):
    return SOS_loss3(torch.view_as_complex(x), torch.sqrt(y), .5, 0,0)

x_f = torchmin.trustregion._minimize_trust_ncg(loss_lam, torch.view_as_real_copy(x_out + 10*torch.randn_like(x_out)), return_all = True, disp=2, max_iter=10)
print(x_f.allvecs[0].shape)