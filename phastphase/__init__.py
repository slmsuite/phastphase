import torch
from torch.fft import fftn, ifftn
import torchmin
from tqdm import tqdm

import warnings

import numpy as np

__version__ = '0.0.1'

def retrieve(
        farfield_data,
        nearfield_support_shape=None,
        **kwargs
    ):
    r"""
    Solves the phase retrieval problem for near-Schwarz objects.

      Given a farfield intensity image :math:`\textbf{y}`,
      find a best-fit complex nearfield image :math:`\textbf{x}`
      such that :math:`\left| \mathcal{F}\{\textbf{x}\} \right|^2 \approx \textbf{y}`.

    Where :math:`\mathcal{F}` is the zero-padded discrete Fourier transform.

    Parameters
    ----------
    farfield_data : torch.Tensor OR numpy.ndarray OR cupy.ndarray OR array_like
        Farfield intensities :math:`\textbf{y}`: the 2D image data to retrieve upon.
    nearfield_support_shape : (int, int) OR None
        The 2D shape of the desired nearfield, the result to retrieve.
        This shape must be smaller than the shape of the farfield
        for the phase retrieval problem to not be underconstrained.
        For best results, the shape should be significantly smaller.
        If ``None``, this is set to half the height and width of the shape of ``farfield_data``.
    **kwargs
        Additional flags to hone retrieval.
        These flags are documented in :meth:`~phastphase.retrieve_()`.

    Returns
    -------
    nearfield_retrieved : torch.Tensor OR numpy.ndarray OR cupy.ndarray

        Recovered complex object :math:`\textbf{x}` which is a best fit for
        :math:`|\mathcal{F}\{\textbf{x}\}|^2 \approx \textbf{y}`.

        - If ``farfield_data`` is a ``torch.Tensor``, then a ``torch.Tensor`` is returned.
        - If ``farfield_data`` is a ``cupy.ndarray`` and the ``device`` is not the CPU,
          then a ``cupy.ndarray`` is returned.
        - Otherwise, a ``numpy.array`` is returned.
    """
    # Grab nearfield amp
    known_nearfield_amp = kwargs.pop("known_nearfield_amp", None)
    support_mask = kwargs.pop("support_mask",None)
    known_nearfield_amp_torch = None

    # Determine the class of the data, and force it to be a torch tensor.
    xp = None   # numpy/cupy module; None ==> torch
    if isinstance(farfield_data, torch.Tensor):
        support_mask_torch = support_mask
        farfield_torch = farfield_data.detach()
        if known_nearfield_amp is not None:
            known_nearfield_amp_torch = known_nearfield_amp.detach()
    else:
        # Determine whether cupy or numpy is used.
        if hasattr(farfield_data, "get_array_module"):
            xp = farfield_data.get_array_module()
        else:
            xp = np

        # Determine the device based on the value of device.
        device = kwargs.pop("device", None)
        if torch.cuda.is_available() and (device is not False):
            if device is not None:
                torch_device = device
            else:
                torch_device = torch.device('cuda')
        else:
            torch_device = torch.device('cpu')

        # as_tensor uses the same memory, including the cupy/GPU case, if possible based on the device.
        try:
            farfield_torch = torch.as_tensor(farfield_data, device=torch_device)
            support_mask_torch = torch.as_tensor(support_mask.copy(), device=torch_device)
            if known_nearfield_amp is not None:
                known_nearfield_amp_torch = torch.as_tensor(known_nearfield_amp, device=torch_device)
        except:
            # torch doesn't support some numpy features such as negative slides, so
            # we fallback to copying the memory.
            farfield_torch = torch.as_tensor(farfield_data.copy(), device=torch_device)
            if known_nearfield_amp is not None:
                known_nearfield_amp_torch = torch.as_tensor(known_nearfield_amp.copy(), device=torch_device)

    # Wrap the actual phase retrieval functions.
    retrieved_object = retrieve_(
        farfield_torch,
        nearfield_support_shape,
        known_nearfield_amp=known_nearfield_amp_torch,
        support_mask = support_mask_torch,
        **kwargs
    )

    # Return the result in the original format.
    if xp is None:
        return retrieved_object
    else:
        return xp.asarray(retrieved_object)

def retrieve_(
        farfield_data: torch.Tensor,
        nearfield_support_shape,
        verbose : bool = False,
        assume_twinning: bool = False,
        farfield_offset: float = 1e-12,
        grad_tolerance: float = 1e-9,
        adam_iters: int = 0,
        cost_reg: float = 1,
        reference_point = None,
        known_nearfield_amp = None,
        use_trust_region = True,
        support_mask = None,
        assume_real : bool = False,
        fast_winding: bool = False,
        tr_max_iter: int = 500
    ) -> torch.Tensor:
    r"""
    Wrapped by :meth:`retrieve` to make the many optional flags less intimidating
    to the casual user.
    See :meth:`retrieve` for high level documentation. :meth:`retrieve` also handles
    conversion of :mod:`torch`/:mod:`numpy`/:mod:`cupy` data, while this function only
    accepts :mod:`torch`.

    Parameters
    ----------
    farfield_data : torch.Tensor
        Farfield intensities :math:`\textbf{y}`: the 2D image data to retrieve upon.
    nearfield_support_shape : (int, int) OR None
        The 2D shape of the desired nearfield, the result to retrieve.
        This shape must be smaller than the shape of the farfield
        for the problem to not be underconstrained.
        For best results, the shape should be significantly smaller.
        If ``None``, this is set to half the height and width of the shape of ``farfield_data``.
    device : torch.device OR bool OR None
        Torch device to use in optimization.
        If ``None`` or ``True``, constructs the current torch device,
        prioritizing CUDA and falling back to the CPU.
        If ``False``, the CPU is forced.
        Defaults to ``None``.
    verbose : bool
        Whether to use :mod:`tqdm` progress bars and print statements.
        Defaults to ``False``.
    assume_twinning : bool
        Flag indicating that image twinning will occur due to loose support.
        Causes the function to attempt to remove the extraneous twin image.
    farfield_offset : float
        Regularization if :math:`Y` has zeros. Defaults to ``1e-12``.
    adam_iters : int
        Number of iterations of the ``AdamW`` algorithm to run before
        trust region super-Newton optimization.
        Defaults to 10.
    grad_tolerance : float
        Gradient tolerance for the trust region super-Newton optimization.
        For large images, the computation of the Hessian-Vector product can hang.
        In this case, the user can set the ``grad_tolerance`` to ``inf`` to avoid this step.
        Defaults to ``1e-9``.
    cost_reg : float
        Regularization parameter to remove the global phase ambiguity in the optimization landscape.  Defaults to ``1``.
    reference_point : (int, int) OR None
        Location of brightest pixel in object (if known) in :math:`(p_y, p_x)` form.
        If ``None``, the winding number of the :math:`z`-transform is used to compute a guess.
    known_nearfield_amp : array_like OR None
        Optional nearfield amplitude constraint to apply.
        Should be the same shape as ``nearfield_support_shape``.

    Returns
    -------
    nearfield_retrieved : torch.Tensor
        Recovered complex object :math:`\textbf{x}` which is a best fit for
        :math:`|\mathcal{F}\{\textbf{x}\}|^2 \approx \textbf{y}`.
    """
    
    # Fix the datatype of the tensor.
    if not farfield_data.dtype in [torch.float32, torch.float64]:
        warnings.warn(
            f"Datatype {farfield_data.dtype } is not compatible with retrieval. "
            "casting to torch.float32."
        )
        farfield_data = farfield_data.to(torch.float32)
    if known_nearfield_amp is not None and not known_nearfield_amp.dtype in [torch.float32, torch.float64]:
        warnings.warn(
            f"Datatype {known_nearfield_amp.dtype } is not compatible with retrieval. "
            "casting to torch.float32."
        )
        known_nearfield_amp = known_nearfield_amp.to(torch.float32)

    # Start by calculating the cepstrum of the image.
    # The farfield_offset is used to avoid log(0).
    y = torch.add(farfield_data, farfield_offset)
    y = y.detach().clone()
    cepstrum = ifftn(torch.log(y))

    # Parse the support and determine the center of the cepstrum.
    tight_support = nearfield_support_shape
    if tight_support is None:
        tight_support = (y.shape[0] // 2, y.shape[1] // 2)
    if reference_point is not None:
        [wind_1, wind_2] = reference_point
    else:
        [wind_1, wind_2] = bisection_winding_calc(tight_support, cepstrum.numpy(force=True))
        if verbose: print(f'Calculated reference location is: {[wind_1,wind_2]}')

    
    # Shift the data based upon the found center.
    mask = torch.zeros_like(cepstrum)
    mask[0:y.shape[0]//2, 0:y.shape[1]//2] = 2
    mask[0:2*wind_1+1, 0:2*wind_2+1] = 1
    mask = torch.roll(mask, (-wind_1, -wind_2), dims=(0, 1))
    filtered_logy = fftn(torch.mul(mask, cepstrum))
    x_out = ifftn(torch.exp(1/2*filtered_logy), norm='ortho')
    x_out = torch.roll(x_out, (wind_1, wind_2), dims=(0, 1))
    x_out = schwarz_transform(cepstrum,(wind_1,wind_2))
    
    # Crop the data based upon the nearfield size.
    x_out = x_out[0:tight_support[0], 0:tight_support[1]]
    x_out = x_out/torch.exp(1j*torch.angle(x_out[wind_1, wind_2]))
    if support_mask is None:
        support_mask = torch.ones_like(x_out)
    x_out *= support_mask
    if assume_twinning:
        x_out = x_out[0:wind_1+1,0:wind_2+1]

    # Construct the figure of merit based on whether a nearfield amp guess was provided.
    if known_nearfield_amp is not None:
        known_nearfield_amp = torch.add(known_nearfield_amp, farfield_offset)
        def loss_lam_L2(x):
            return SOS_loss(
                support_mask*torch.view_as_complex(x), torch.sqrt(y),
                cost_reg, wind_1, wind_2,
                known_nearfield_amp, amp_lambda=1
            )
    else:
        def loss_lam_L2(x):
            return SOS_loss2(support_mask*torch.view_as_complex(x), torch.sqrt(y), cost_reg, wind_1, wind_2)

    if fast_winding:
        return(x_out, loss_lam_L2(x_out), (wind_1, wind_2))
    # Start with an Adam optimization.
    x0 = torch.view_as_real_copy(x_out)
    x0.requires_grad_()
    optimizer = torch.optim.AdamW([x0], lr=.01, foreach=True)
    iterator = range(adam_iters)
    if verbose and adam_iters > 1: iterator = tqdm(iterator, desc="Adam")
    for _ in iterator:
        optimizer.zero_grad()
        loss = loss_lam_L2(x0)
        loss.backward()

        if verbose and adam_iters > 1:
            iterator.set_description(f"Adam (loss={loss}, grad={loss.grad})")

        optimizer.step()


    # Finish with a super-Newton refinement.
    x0 = x0.detach().clone()
    if np.isinf(grad_tolerance):
        x_final = x0
    else:
        if verbose:
            display = 2
        else:
            display = 0
        if use_trust_region:
            result = torchmin.trustregion._minimize_trust_ncg(
                loss_lam_L2,
                x0,
                gtol=grad_tolerance,
                disp = display,
                max_iter = tr_max_iter
            )
        else:
            result = torchmin.bfgs._minimize_lbfgs(
                loss_lam_L2,
                x0,
                gtol=grad_tolerance,
                xtol = 1e-15,
                disp = display
            )
        x_final = result.x


    return torch.view_as_complex_copy(x_final)

def bisection_winding_calc(shape, cepstrum, num_loops=100, verbose=False):
    """
    Calculates the ``reference_point`` by centering the winding.

    Parameters
    ----------
    shape :
        Tight bounding box for object, ``nearfield_support_shape``.
    cepstum: ndarray
        FFT of the log of the data.
    num_loops : int
        Number of iterations to optimize.
    verbose : bool
        Whether to print :mod:`tqdm` progress.

    Returns
    -------
    reference_point : (int, int)
        Centered reference point.
    """
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
def schwarz_transform(cepstrum, winding):
    sz = cepstrum.shape
    mask = torch.zeros_like(cepstrum)
    (n,m) = torch.meshgrid(torch.fft.ifftshift(torch.linspace(-np.ceil(sz[0]/2), sz[0]//2 ,steps=sz[0])),
                           torch.fft.ifftshift(torch.linspace(-np.ceil(sz[1]/2), sz[1]//2 ,steps=sz[1])), indexing='ij')
    if winding == (0,0):
        mask[0:sz[0]//2, 0:sz[1]//2] = 2
        mask[0,0] =1
    elif winding[0] == 0:
         mask[0:sz[0]//2, 0:sz[1]] = 2
         mask[0,0] =1
    elif winding[1] == 0:
        mask[0:sz[0], 0:sz[1]//2] = 2
        mask[0,0] =1
    else:
        mask[winding[0]*n + winding[1]*m >= 0] = 2
        mask = torch.roll(mask, (winding[0], winding[1]), dims=(0, 1))
        mask[0:2*winding[0]+1, 0:2*winding[1]+1]=1
        mask = torch.roll(mask, (-winding[0], -winding[1]), dims=(0, 1))
    filtered_logy = fftn(torch.mul(mask, cepstrum))
    x_out = ifftn(torch.exp(1/2*filtered_logy), norm='ortho')
    x_out = torch.roll(x_out, (winding[0], winding[1]), dims=(0, 1))
    return x_out

def SOS_loss(
    x: torch.Tensor,
    ysqrt: torch.Tensor,
    reg_lambda: float,
    ind1: int,
    ind2: int,
    known_nearfield_amp: torch.Tensor,
    amp_lambda : float
) -> torch.Tensor:
    """Sums of Squares loss function with known nearfield.

    Includes regularization to break global phase symmetry.

    Parameters
    ----------
    x : torch.Tensor
        Near field object.
    y : torch.Tensor
        Far-field Intensities.
    reg_lambda : float
        Regularizer to fix phase.
    amp_lambda : float
        To motivate amplitude constraint.
    ind1 : int
        Reference point index 1.
    ind2 : int
        Reference point index 2.
    known_nearfield_amp : torch.Tensor
        Known amplitude to add as a constraint.

    Returns
    -------
    loss : torch.Tensor
        Loss for this guess ``x``.
    """
    return (
        (torch.square(
            torch.linalg.vector_norm(
                torch.addcdiv(
                    ysqrt,
                    torch.abs(
                        torch.square(fftn(x, s=ysqrt.shape, norm='ortho'))
                    ),
                    ysqrt,
                    value=-1
                )
            )
        ))/8
        + reg_lambda*torch.square(torch.imag(x[ind1, ind2]))/2
        + amp_lambda*(torch.square(
            torch.linalg.vector_norm(
                torch.abs(x) - known_nearfield_amp
            )
        ))/8
    )

def SOS_loss2(
    x: torch.Tensor,
    ysqrt: torch.Tensor,
    reg_lambda: float,
    ind1: int,
    ind2: int
) -> torch.Tensor:
    """Sums of Squares loss function for Phase Retrieval.

    Includes regularization to break global phase symmetry.

    Parameters
    ----------
    x : torch.Tensor
        Near field object.
    y : torch.Tensor
        Far-field Intensities.
    reg_lambda : float
        Regularizer to fix phase.
    ind1 : int
        Reference point index 1.
    ind2 : int
        Reference point index 2.
    known_nearfield_amp : torch.Tensor
        Known amplitude to add as a constraint.

    Returns
    -------
    loss : torch.Tensor
        Loss for this guess ``x``.
    """
    return (
        (torch.square(
            torch.linalg.vector_norm(
                torch.addcdiv(
                    ysqrt,
                    torch.abs(torch.square(fftn(x, s=ysqrt.shape, norm='ortho'))),
                    ysqrt,
                    value=-1
                )
            )
        ))/8
        + reg_lambda*torch.square(torch.imag(x[ind1,ind2]))/2
    )

def real_masked_loss(
    x: torch.Tensor,
    ysqrt: torch.Tensor,
    reg_lambda: float,
    mask: torch.Tensor
) -> torch.Tensor:
    
        return (
        (torch.square(
            torch.linalg.vector_norm(
                mask*torch.addcdiv(
                    ysqrt,
                    torch.abs(torch.square(fftn(x, s=ysqrt.shape, norm='ortho'))),
                    ysqrt,
                    value=-1
                )
            )
        ))/8
        + reg_lambda*torch.square(torch.linalg.vector_norm(torch.imag(x)))/2
    )

