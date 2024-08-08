import torch
from torch.fft import fftn, ifftn
import torchmin


from typing import List

import numpy as np

__version__ = '0.0.1'


def retrieve_phase(
        far_field_data,
        near_field_support,
        gpu=None,
        **kwargs
    ):
    """
    Solves the phase retrieval problem for near-Schwarz objects.

    Parameters
    ----------
        far_field_data : torch.Tensor OR array_like
            Far-field intensity :math:`y`. This can be any array-like data e.g. :mod:`numpy`,
            but passing a :class:`torch.Tensor` is suggested to reduce allocation overhead.
        near_field_support : (int, int)
            TODO. Is there some default value that could be given?
        gpu : torch.device OR None OR bool
            Torch device to use in optimization.
            If ``None`` or ``True``, constructs the current torch device, prioritizing CUDA and
            falling back to the CPU.
            If ``False``, the CPU is forced.
        **kwargs
            Additional flags to hone retrieval.

            -   ``"assume_twinning"`` - bool

                TODO

            -   ``"far_field_offset"`` - float

                Regularization if y has zeros. Defaults to ``1e-12``.

            -   ``"grad_tolerance"`` - float:

                Gradient tolerance for the Super-Newton optimization. Defaults to ``1e-9``.

            -   ``"adam_iters"`` - int

                Number of iterations of the ``AdamW`` algorithm to run before Super-Newton.
                Defaults to 100.

            -   ``"cost_reg"``: float = 1,

                TODO

            -   ``"reference_point"``: (int, int) OR None

                TODO

            -   ``"known_mags"`` - array_like OR None

                TODO

    Returns
    -------
        x_final : torch.Tensor OR numpy.ndarray
            Recovered complex object :math:`x`.

            - If ``y_data`` is a torch Tensor, then a Tensor is returned.
            - Otherwise, a numpy array is returned.

        TODO: as a small comment, the function is named ``retreive_phase``, but the complex
        data is returned.
    """

    if isinstance(far_field_data, torch.Tensor):
        use_numpy = False
        far_field_torch = far_field_data.clone().detach()
    else:
        use_numpy = True
        far_field_data = far_field_data.astype(np.float64)

        if torch.cuda.is_available() and gpu is not False:
            if gpu is not None:
                torch_device = gpu
            else:
                torch_device = torch.device('cuda')
            far_field_torch = torch.from_numpy(far_field_data).cuda(device=torch_device)
        else:
             far_field_torch = torch.from_numpy(far_field_data).clone().detach()

    retrieved_object = _retrieve_phase(
        far_field_torch,
        near_field_support,
        near_field_support,
        **kwargs
    )

    if use_numpy:
        return retrieved_object.numpy(force=True)
    else:
        return retrieved_object


def _retrieve_phase(
        y_data: torch.Tensor,
        tight_support: List[int],
        loose_support: List[int],           # TODO: unused?
        assume_twinning: bool = False,
        far_field_offset: float = 1e-12,
        grad_tolerance: float = 1e-9,
        adam_iters: int = 0,
        cost_reg: float = 1,
        reference_point = None,
        known_mags = None,
    ) -> torch.Tensor:
    """
    Wrapped by :meth:`retrieve_phase` to make the many flags less intimidating.
    See :meth:`retrieve_phase` for documentation. :meth:`retrieve_phase` also handles
    conversion of numpy/torch/cupy
    """

    y = torch.add(y_data, far_field_offset)
    cepstrum = ifftn(torch.log(y))
    if reference_point is not None:
        [wind_1, wind_2] = reference_point
    else:
        [wind_1, wind_2] = bisection_winding_calc(tight_support, cepstrum.numpy(force=True))
    print(f'Calculated reference location is: {[wind_1,wind_2]}')
    mask = torch.zeros_like(cepstrum)
    mask[0:y.shape[0]//2, 0:y.shape[1]//2] = 2
    mask = torch.roll(mask, (-wind_1, -wind_2), dims=(0, 1))
    mask[0, 0] = 1
    filtered_logy = fftn(torch.mul(mask, cepstrum))
    x_out = ifftn(torch.exp(1/2*filtered_logy), norm='ortho')
    x_out = torch.roll(x_out, (wind_1, wind_2), dims=(0, 1))

    x_out = x_out[0:tight_support[0], 0:tight_support[1]]
    x_out = x_out/torch.exp(1j*torch.angle(x_out[wind_1, wind_2]))
    if known_mags is not None:
        def loss_lam_L2(x): return SOS_loss(
                    torch.view_as_complex(x), torch.sqrt(y),cost_reg, wind_1, wind_2, known_mags)
    else:
        def loss_lam_L2(x): return SOS_loss2(
                    torch.view_as_complex(x), torch.sqrt(y),cost_reg, wind_1, wind_2)

    if assume_twinning:
        x_out = x_out[0:wind_1+1,0:wind_2+1]
    x0 = torch.view_as_real_copy(x_out)
    x0.requires_grad_()
    optimizer = torch.optim.AdamW([x0], lr=.01, foreach=True)
    for _ in range(adam_iters):
        optimizer.zero_grad()
        loss = loss_lam_L2(x0)
        loss.backward()
        optimizer.step()


    x0 = x0.detach().clone()
    result = torchmin.trustregion._minimize_trust_ncg(loss_lam_L2, x0, gtol=grad_tolerance,disp=2, max_trust_radius = 1e3, initial_trust_radius=100)

    x_final = result.x

    return torch.view_as_complex_copy(x_final)

def bisection_winding_calc(shape, cepstrum, num_loops=100):
    #shape : tight bounding box for object
    #cepstum: ndarray of cepstrum
    #threshbounds, tuple of (high,low)
    n = shape[0]
    m = shape[1]
    thresh_bounds = (np.max(np.abs(cepstrum)), np.min(np.abs(cepstrum)))
    offdiagorth = np.abs(cepstrum[0:n//2,-m:])
    primary_orthant = np.abs(cepstrum[0:n,0:m])
    thresh_high = thresh_bounds[0]
    thresh_low=thresh_bounds[1]
    for i in range(num_loops):
        threshhold=(thresh_high+thresh_low)/2.
        height = np.where(np.heaviside(offdiagorth-threshhold,0)[:,-m:-m//2+1])[0].max(initial=0) + np.where(np.heaviside(primary_orthant-threshhold,0))[0].max(initial=0) +1
        winding_num_1 = np.where(np.heaviside(offdiagorth-threshhold,0)[:,-m:-m//2+1])[0].max(initial=0)

        if height > n:
            thresh_low = threshhold
        elif height < n:
            thresh_high = threshhold
        elif height == n:
            break

    offdiagorth = np.abs(cepstrum[-n:,0:m//2+1])
    thresh_high = thresh_bounds[0]
    thresh_low=thresh_bounds[1]
    for j in range(num_loops):
        threshhold=(thresh_high+thresh_low)/2.
        width = np.where(np.heaviside(offdiagorth-threshhold,0)[-n:-n//2+1,0:m])[1].max(initial=0) + np.where(np.heaviside(primary_orthant-threshhold,0))[1].max(initial=0) +1
        winding_num_2 = np.where(np.heaviside(offdiagorth-threshhold,0)[-n:-n//2+1,0:m])[1].max(initial=0)
        if width > m:
            thresh_low = threshhold
        elif width < m:
            thresh_high = threshhold
        elif width == m:
            break

    return (winding_num_1, winding_num_2)

def SOS_loss(x: torch.Tensor, ysqrt: torch.Tensor,reg_lambda: float, ind1: int, ind2: int, known_mags: torch.Tensor) -> torch.Tensor:
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
    return (torch.square(torch.linalg.vector_norm(torch.addcdiv(ysqrt,torch.abs(torch.square(fftn(x, s=ysqrt.shape, norm='ortho'))),ysqrt,value=-1))))/8 \
            + reg_lambda*torch.square(torch.imag(x[ind1, ind2]))/2 + (torch.square(torch.linalg.vector_norm(torch.addcdiv(known_mags,torch.square(torch.abs(x)),known_mags,value=-1))))/8

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
