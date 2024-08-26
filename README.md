<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/slmsuite/phastphase/main/docs/source/static/phastphase-dark.svg">
<img alt="phastphase" src="https://raw.githubusercontent.com/slmsuite/phastphase/main/docs/source/static/phastphase.svg" width="256">
</picture>
</p>

<h2 align="center">Fast Phase Retrieval for Near-Schwarz Objects</h2>

<p align="center">
<a href="https://phastphase.readthedocs.io/en/latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/phastphase/badge/?version=latest"></a>
</p>

`phastphase` is a GPU-accelerated implementation of the Fast Phase Retrieval algorithm for solving the support-constrained phase retrieval problem.

> Given a farfield intensity image $\textbf{y}$, find a best-fit complex nearfield image $\textbf{x}$ such that $\left| \mathcal{F}\{\textbf{x}\} \right|^2 \approx \textbf{y}$.
    
Where $\mathcal{F}$ is the zero-padded discrete Fourier transform.
Near-Schwarz objects are defined by the phase of their Z-Transform:
    $$ |\text{Arg}(X(\textbf{z})) - \text{Arg}(\textbf{z}^\textbf{n})| \leq \frac{\pi}{2} $$
While the algorithm may work for objects outside of this class, it is only proven to work for near-Schwarz objects. 

## Installation

Install the stable version of `phastphase` from [PyPI](https://pypi.org/project/phastphase/) using:

```console
pip install phastphase
```

Install the latest version of `phastphase` from GitHub using:

```console
pip install git+https://github.com/slmsuite/phastphase
```