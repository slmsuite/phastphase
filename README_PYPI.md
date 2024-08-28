<p align="center">
<img alt="phastphase" src="https://raw.githubusercontent.com/slmsuite/phastphase/main/docs/source/static/phastphase.svg" width="256">
</p>

<h2 align="center">Phase Retrieval For Near Schwarz Objects</h2>

<p align="center">
<a href="https://phastphase.readthedocs.io/en/latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/phastphase/badge/?version=latest"></a>
<a href="https://arxiv.org/abs/2407.01350"><img alt="Citation" src="https://img.shields.io/badge/cite-arXiv%3A2407.01350-B31B1B.svg"></a>
</p>
'phastphase' is a gpu-accelerated implementation of the Fast Phase Retrieval algorithm for solving the support-constrained Phase Retrieval problem on near-Schwarz objects. 

Given far-field intensities $\textbf{y}$ the algorithm returns a near-Schwarz object satisfying:
$$|\mathcal{F}\{\textbf{x}\}|^2 \approx \textbf{y}$$
Where $\mathcal{F}$ is the zero-padded DFT.

Near-Schwarz objects are defined by the phase of their Z-Transform:
    $$ |\text{Arg}(X(\textbf{z})) - \text{Arg}(\textbf{z}^\textbf{n})| \leq \frac{\pi}{2} $$
While the algorithm may work for objects outside of this class, it is only proven to work for near-Schwarz objects. 
## Installation

Install the stable version of `phastphase` from [PyPI](https://pypi.org/project/phastphase/) using:

```console
$ pip install phastphase
```

Install the latest version of `phastphase` from GitHub using:

```console
$ pip install git+https://github.com/slmsuite/phastphase
```