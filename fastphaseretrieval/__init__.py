import torch
from torch.fft import fftn, ifftn
import torchmin
from tqdm import tqdm

from typing import List

import numpy as np

__version__ = '0.0.1'


def retrieve_phase(
        farfield_data,
        nearfield_support_shape=None,
        gpu=None,
        verbose=False,
        **kwargs
    ):
    r"""
    Solves the phase retrieval problem for near-Schwarz objects.
    This finds a best-fit complex nearfield :math:`x` 
    given a farfield intensity :math:`Y \approx |y|^2`, where :math:`y = \mathcal{F}\{x\}`.

    Parameters
    ----------
        farfield_data : torch.Tensor OR array_like
            Farfield intensity :math:`|y|^2`: the 2D image data to retrieve upon.
            This can be any array-like data e.g. :mod:`numpy`,
            but passing a :class:`torch.Tensor` is suggested to reduce allocation overhead.
        nearfield_support_shape : (int, int) OR None
            The 2D shape of the desired nearfield, the result to retrieve.
            This shape must be smaller than the shape of the farfield
            for the problem to not be underconstrained.
            For best results, the shape should be significantly smaller.
            If ``None``, this is set to half the height and width of the farfield shape.
        gpu : torch.device OR bool OR None
            Torch device to use in optimization.
            If ``None`` or ``True``, constructs the current torch device, 
            prioritizing CUDA and falling back to the CPU.
            If ``False``, the CPU is forced.
        verbose : bool
            Whether to use :mod:`tqdm` progress bars and print statements.
        **kwargs
            Additional flags to hone retrieval:

            -   ``"assume_twinning"`` - bool

                Flag indicating that image twinning will occur due to loose support. 
                Causes the function to attempt to remove the extraneous twin image. 

            -   ``"farfield_offset"`` - float

                Regularization if :math:`Y` has zeros. Defaults to ``1e-12``.

            -   ``"adam_iters"`` - int

                Number of iterations of the ``AdamW`` algorithm to run before Trust Region Newton.
                Defaults to 100.

            -   ``"grad_tolerance"`` - float

                Gradient tolerance for the Trust Region Optimization. Defaults to ``1e-9``.

            -   ``"cost_reg"`` - float

                Regularization parameter to remove the global phase ambiguity in the optimization landscape.  Defaults to ''1''. 

            -   ``"reference_point"`` - (int, int) OR None

                Location of brightest pixel in object (if known) in :math:`(p_y, p_x)` form.
                Otherwise if ``None``, the winding number of the :math:`z`-transform is used to compute a guess.

            -   ``"known_nearfield_amp"`` - array_like OR None

                Optional nearfield amplitude constraint to apply.
                Should be the same shape as ``nearfield_support_shape``

    Returns
    -------
        nearfield_retrieved : torch.Tensor OR numpy.ndarray OR 
            Recovered complex object :math:`x` which is a best fit for :math:`|\mathcal\{x\}|^2 \approx Y`.

            - If ``y_data`` is a ``torch.Tensor``, then a ``torch.Tensor`` is returned.
            - Otherwise, a ``numpy.array`` is returned.

        TODO: as a small comment, the function is named ``_retrieve_phase``, but the complex
        data is returned.
    """
    # Determine the class of the data, and force it to be a torch tensor.
    xp = None   # numpy/cupy module; None ==> torch
    if isinstance(farfield_data, torch.Tensor):
        farfield_torch = farfield_data.detach()
    else:
        # Determine whether cupy or numpy is used.
        if hasattr(farfield_data, "get_array_module"):
            xp = farfield_data.get_array_module()
        else:
            xp = np
        
        # Determine the device based on the value of gpu.
        if torch.cuda.is_available() and (gpu is not False):
            if gpu is not None:
                torch_device = gpu
            else:
                torch_device = torch.device('cuda')
        else:
            torch_device = torch.device('cpu')
        
        # as_tensor uses the same memory, including the cupy/GPU case, if possible based on the device.
        try:
            farfield_torch = torch.as_tensor(farfield_data, device=torch_device)
        except:
            # torch doesn't support some numpy features such as negative slides, so
            # we fallback to copying the memory.
            farfield_torch = torch.as_tensor(farfield_data.copy(), device=torch_device)

    # Wrap the actual phase retrieval functions.
    retrieved_object = _retrieve_phase(
        farfield_torch,
        nearfield_support_shape,
        verbose,
        **kwargs
    )

    # Return the result in the original format.
    if xp is None:
        return retrieved_object
    else:
        return xp.asarray(retrieved_object)

def _retrieve_phase(
        y_data: torch.Tensor,
        tight_support: List[int],
        verbose : bool = False,
        assume_twinning: bool = False,
        farfield_offset: float = 1e-12,
        grad_tolerance: float = 1e-9,
        adam_iters: int = 0,
        cost_reg: float = 1,
        reference_point = None,
        known_nearfield_amp = None,
    ) -> torch.Tensor:
    """
    Wrapped by :meth:`retrieve_phase` to make the many flags less intimidating.
    See :meth:`retrieve_phase` for documentation. :meth:`retrieve_phase` also handles
    conversion of numpy/torch/cupy
    """
    # Start by calculating the sepstrum of the image.
    # The farfield_offset is used to avoid log(0).
    y = torch.add(y_data, farfield_offset)
    cepstrum = ifftn(torch.log(y))

    # Determine the center of the cepstrum.
    if reference_point is not None:
        [wind_1, wind_2] = reference_point
    else:
        [wind_1, wind_2] = bisection_winding_calc(tight_support, cepstrum.numpy(force=True))
        if verbose: print(f'Calculated reference location is: {[wind_1,wind_2]}')

    # Shift the data based upon the found center.
    mask = torch.zeros_like(cepstrum)
    mask[0:y.shape[0]//2, 0:y.shape[1]//2] = 2
    mask = torch.roll(mask, (-wind_1, -wind_2), dims=(0, 1))
    mask[0, 0] = 1
    filtered_logy = fftn(torch.mul(mask, cepstrum))
    x_out = ifftn(torch.exp(1/2*filtered_logy), norm='ortho')
    x_out = torch.roll(x_out, (wind_1, wind_2), dims=(0, 1))

    # Crop the data based upon the nearfield size.
    x_out = x_out[0:tight_support[0], 0:tight_support[1]]
    x_out /= torch.exp(1j*torch.angle(x_out[wind_1, wind_2]))
    if assume_twinning:
        x_out = x_out[0:wind_1+1,0:wind_2+1]

    # Construct the figure of merit based on whether a nearfield amp guess was provided.
    if known_nearfield_amp is not None:
        def loss_lam_L2(x): 
            return SOS_loss(torch.view_as_complex(x), torch.sqrt(y), cost_reg, wind_1, wind_2, known_nearfield_amp)
    else:
        def loss_lam_L2(x): 
            return SOS_loss2(torch.view_as_complex(x), torch.sqrt(y), cost_reg, wind_1, wind_2)

    # Start with an Adam optimization.
    x0 = torch.view_as_real_copy(x_out)
    x0.requires_grad_()
    optimizer = torch.optim.AdamW([x0], lr=.01, foreach=True)
    for _ in range(adam_iters):
        optimizer.zero_grad()
        loss = loss_lam_L2(x0)
        loss.backward()
        optimizer.step()

    # Finish with a super-Newton refinement.
    x0 = x0.detach().clone()
    result = torchmin.trustregion._minimize_trust_ncg(
        loss_lam_L2, 
        x0, 
        gtol=grad_tolerance,
        disp=2, 
        max_trust_radius=1e3, 
        initial_trust_radius=100
    )
    x_final = result.x

    return torch.view_as_complex_copy(x_final)

def bisection_winding_calc(shape, cepstrum, num_loops=100, verbose=False):
    #shape : tight bounding box for object
    #cepstum: ndarray of cepstrum
    #threshbounds, tuple of (high,low)
    n = shape[0]
    m = shape[1]
    thresh_bounds = (np.max(np.abs(cepstrum)), np.min(np.abs(cepstrum)))
    offdiagorth = np.abs(cepstrum[0:n//2,-m:])
    primary_orthant = np.abs(cepstrum[0:n,0:m])
    thresh_high = thresh_bounds[0]
    thresh_low = thresh_bounds[1]
    iterator = range(num_loops)
    if verbose: iterator = tqdm(iterator, desc="X winding")
    for _ in iterator:
        threshhold = (thresh_high+thresh_low)/2.
        height = (
            np.where(np.heaviside(offdiagorth-threshhold,0)[:,-m:-m//2+1])[0].max(initial=0) + 
            np.where(np.heaviside(primary_orthant-threshhold,0))[0].max(initial=0) +
            1
        )
        winding_num_1 = np.where(np.heaviside(offdiagorth-threshhold,0)[:,-m:-m//2+1])[0].max(initial=0)

        if height > n:
            thresh_low = threshhold
        elif height < n:
            thresh_high = threshhold
        elif height == n:
            break

    offdiagorth = np.abs(cepstrum[-n:,0:m//2+1])
    thresh_high = thresh_bounds[0]
    thresh_low = thresh_bounds[1]
    iterator = range(num_loops)
    if verbose: iterator = tqdm(iterator, desc="Y winding")
    for _ in iterator:
        threshhold = (thresh_high+thresh_low)/2.
        width = (
            np.where(np.heaviside(offdiagorth-threshhold,0)[-n:-n//2+1,0:m])[1].max(initial=0) + 
            np.where(np.heaviside(primary_orthant-threshhold,0))[1].max(initial=0) +
            1
        )
        winding_num_2 = np.where(np.heaviside(offdiagorth-threshhold,0)[-n:-n//2+1,0:m])[1].max(initial=0)

        if width > m:
            thresh_low = threshhold
        elif width < m:
            thresh_high = threshhold
        elif width == m:
            break

    return (winding_num_1, winding_num_2)

def SOS_loss(x: torch.Tensor, ysqrt: torch.Tensor,reg_lambda: float, ind1: int, ind2: int, known_nearfield_amp: torch.Tensor) -> torch.Tensor:
    """Sums of Squares loss function for Phase Retrieval.

    Includes regularization to break global phase symmetry.


    Args:
        x (torch.Tensor): Near field object.
        y (torch.Tensor): Far-field Intensities
        reg_;a,bda (float): Regularizer to fix phase.
        ind1 (int): index_1
        ind2 (int): index_2

    Returns:
        torch.Tensor: Loss
    """
    return (torch.square(torch.linalg.vector_norm(torch.addcdiv(ysqrt, torch.abs(torch.square(fftn(x, s=ysqrt.shape, norm='ortho'))), ysqrt, value=-1))))/8 \
            + reg_lambda*torch.square(torch.imag(x[ind1, ind2]))/2 + (torch.square(torch.linalg.vector_norm(torch.addcdiv(known_nearfield_amp,torch.square(torch.abs(x)),known_nearfield_amp,value=-1))))/8

def SOS_loss2(x: torch.Tensor, ysqrt: torch.Tensor,reg_lambda: float, ind1: int, ind2: int) -> torch.Tensor:
    """Sums of Squares loss function for Phase Retrieval.

    Includes regularization to break global phase symmetry.


    Args:
        x (torch.Tensor): Near field object.
        y (torch.Tensor): Far-field Intensities
        reg_;a,bda (float): Regularizer to fix phase.
        ind1 (int): index_1
        ind2 (int): index_2

    Returns:
        torch.Tensor: Loss
    """
    return (torch.square(torch.linalg.vector_norm(torch.addcdiv(ysqrt,torch.abs(torch.square(fftn(x, s=ysqrt.shape, norm='ortho'))),ysqrt,value=-1))))/8 + reg_lambda*torch.square(torch.imag(x[ind1,ind2]))/2
