# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The trust-region minimization algorithm."""
from functools import partial
from typing import Callable, NamedTuple, Type, Union, Optional, Tuple

import jax.numpy as jnp
import jax.numpy.linalg as jnpla

from jax import lax, jit, jvp, value_and_grad, grad
from ._quad_subproblem import CGSteihaugSubproblem


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


# simple class to store values in each optimization step
class _TrustRegionResults(NamedTuple):
  converged: Union[bool, jnp.ndarray]
  good_approx: Union[bool, jnp.ndarray]
  k: Union[int, jnp.ndarray]
  x_k: jnp.ndarray 
  f_k: Union[float, jnp.ndarray]
  g_k: jnp.ndarray
  g_k_mag: jnp.ndarray
  nfev: Union[int, jnp.ndarray]
  ngev: Union[int, jnp.ndarray]
  trust_radius: Union[float, jnp.ndarray]
  status: Union[int, jnp.ndarray]


def minimize_trust_region(
    fun: Callable,
    x0: jnp.ndarray,
    maxiter: Optional[int] = None,
    norm=jnp.inf,
    gtol: float = 1e-5,
    max_trust_radius: Union[float, jnp.ndarray] = 1000.,
    initial_trust_radius: Union[float, jnp.ndarray] = 1.0,
    eta: Union[float, jnp.ndarray] = 0.15,
    method = "trust-ncg",
) -> _TrustRegionResults:

  if not (0 <= eta < 0.25):
    raise Exception("invalid acceptance stringency")
  if gtol < 0.:
    raise Exception("gradient tolerance must be positive")
  if max_trust_radius <= 0:
    raise Exception("max trust radius must be positive")
  if initial_trust_radius <= 0:
    raise ValueError("initial trust radius must be positive")
  if initial_trust_radius >= max_trust_radius:
    raise ValueError("initial trust radius must be less than the max trust radius")

  if method == "trust-ncg":
    subp = CGSteihaugSubproblem
  else:
    raise ValueError("Method {} not recognized".format(method))
  if maxiter is None:
    maxiter = jnp.size(x0) * 200

  vg_f = value_and_grad(fun)
  g_f = grad(fun)
  f_0, g_0 = vg_f(x0)

  init_params = _TrustRegionResults(
      converged=False,
      good_approx=jnp.isfinite(jnpla.norm(g_0, ord=norm)),
      k=1,
      x_k=x0,
      f_k=f_0,
      g_k=g_0,
      g_k_mag=jnpla.norm(g_0, ord=norm),
      nfev=1,
      ngev=1,
      trust_radius=initial_trust_radius,
      status=0
  )

  # function to generate the hessian vector product function
  def _hvp(g_f, primals, tangents):
    return jvp(g_f, (primals,), (tangents,))[1]

  # condition for the main trust region optimization loop
  def _trust_region_cond_f(params: _TrustRegionResults) -> bool:
    return (jnp.logical_not(params.converged)
        & (params.k < maxiter)
        & params.good_approx)

  # function to take a constrained gradient step or adjust trust region size for next iteration
  def _trust_region_body_f(params: _TrustRegionResults) -> _TrustRegionResults:
    # compute a new hessian vector product function given the current state
    hessvp = partial(_hvp, g_f, params.x_k)

    # we should add a interal success check for future subp approaches that might not be solvable
    # (e.g., non-PSD hessian)
    result = subp(
        params.f_k,
        params.g_k,
        params.g_k_mag,
        hessvp,
        params.trust_radius,
        norm=norm
    )

    pred_f_kp1 = result.pred_f
    x_kp1 = params.x_k + result.step
    f_kp1, g_kp1 = vg_f(x_kp1)

    delta = params.f_k - f_kp1
    pred_delta = params.f_k - pred_f_kp1
  
    # update the trust radius according to the actual/predicted ratio
    # use `where` to avoid branching. this is a simple scalar check so not much computational overhead
    rho = delta / pred_delta
    tr = params.trust_radius
    cur_tradius = jnp.where(rho < 0.25, tr * 0.25, tr)
    cur_tradius = jnp.where((rho > 0.75) & result.hits_boundary, jnp.minimum(2. * tr, max_trust_radius), cur_tradius)
  
    # compute norm to check for convergence
    g_kp1_mag = jnpla.norm(g_kp1, ord=norm)

    # if the ratio is high enough then accept the proposed step
    # repeated check to skirt using cond/branching
    f_kp1 = jnp.where(rho > eta, f_kp1, params.f_k)
    x_kp1 = jnp.where(rho > eta, x_kp1, params.x_k)
    g_kp1 = jnp.where(rho > eta, g_kp1, params.g_k)
    g_kp1_mag = jnp.where(rho > eta, g_kp1_mag, params.g_k_mag)
  
    iter_params = _TrustRegionResults(
        converged=g_kp1_mag < gtol,
        good_approx=pred_delta > 0,
        k=params.k + 1,
        x_k=x_kp1,
        f_k=f_kp1,
        g_k=g_kp1,
        g_k_mag=g_kp1_mag,
        nfev=params.nfev + result.nfev + 1, 
        ngev=params.ngev + result.ngev + 1, 
        trust_radius=cur_tradius,
        status=params.status
    )

    return iter_params

  state = lax.while_loop(_trust_region_cond_f, _trust_region_body_f, init_params)
  status = jnp.where(
      state.converged,
      0,  # converged
      jnp.where(
          state.k == maxiter,
          1,  # max iters reached
          jnp.where(
              state.good_approx,
              -1,   # undefined
              2,   # poor approx
          )
      )
  )
  state = state._replace(status=status)

  return state