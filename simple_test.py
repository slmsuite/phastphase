import scipy.optimize
import torch
from torch.fft import fftn
import numpy as np
from phastphase import retrieve, SOS_loss2, SOS_loss3, masked_loss
from phastphase.script_modules import create_wind_eval
import scipy
if torch.cuda.is_available():
    tensor_device = torch.device('cpu')
else:
    tensor_device = torch.device('cpu')

p = 6                       #power of 2 for the near field image
overs = 4                   #oversampling
N = 2**p                    #number of pixels in 1 dimension
x = torch.randn((N,N),
                dtype=torch.cdouble, 
                device = tensor_device
                )

mask = np.ones((N,N))
d = 1
x[d,d] =  .9j*N
y = torch.square(torch.abs(fftn(x,(overs*N, overs*N), norm = 'ortho')))
x_out = retrieve(y, [N,N],grad_tolerance = 1e-9,reference_point = [d,d], verbose=False,tr_max_iter = 1000, oversample_ratio=2, adam_iters=0)
print(torch.linalg.vector_norm(x_out - x)/N**2)

mask = torch.zeros_like(y)
mask[0:2:, 0:2:] = 1
mask = 1-mask
xshape = torch.view_as_real_copy(x_out)
x0 = torch.flatten(xshape).numpy(force=True)
def loss_func(x): return masked_loss(torch.view_as_complex(x.view_as(xshape)), torch.sqrt(y),1,mask, d,d) 


loss_grad = torch.func.grad(loss_func)

def hvp(x, p):
    return torch.autograd.functional.vhp(loss_func, x, p)[1].t()


def scipy_loss(x):
    torch_in = torch.from_numpy(x)
    return loss_func(torch_in).numpy(force=True)

def scipy_grad(x):
    torch_in = torch.from_numpy(x)
    return loss_grad(torch_in).numpy(force=True)
def scipy_hvp(x,p):
    torch_x = torch.from_numpy(x)
    torch_p = torch.from_numpy(p)
    return hvp(torch_x, torch_p).numpy(force=True)

output = scipy.optimize.minimize(scipy_loss, x0, method = 'trust-krylov', jac = scipy_grad, hessp=scipy_hvp, options={'disp':False,'gtol':1e-7})
print(scipy_loss(output.x))

mask[:,:] = 1
output = scipy.optimize.minimize(scipy_loss, output.x, method = 'trust-ncg', jac = scipy_grad, hessp=scipy_hvp, options={'disp':False})

print(scipy_loss(output.x))
xf = torch.view_as_complex(torch.as_tensor(output.x).view_as(xshape))

print(torch.linalg.vector_norm(xf - x)/N**2)
#output = scipy.optimize.basinhopping(scipy_loss, x0,stepsize=1,niter=500,interval=10, T = 0,minimizer_kwargs={'method':'TNC', 'jac' : scipy_grad, 'hessp' : scipy_hvp, 'tol': 1e-7, 'options': {'maxiter': 1000}},disp=True )
