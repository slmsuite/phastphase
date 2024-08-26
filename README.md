<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/slmsuite/phastphase/main/docs/source/static/phastphase-dark.svg">
<img alt="slmsuite" src="https://raw.githubusercontent.com/slmsuite/phastphase/main/docs/source/static/phastphase.svg" width="256">
</picture>
</p>

<h2 align="center">Phase Retrieval For Near Schwarz Objects</h2>

<p align="center">
<a href="https://phastphase.readthedocs.io/en/latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/phastphase/badge/?version=latest"></a>
</p>
'phastphase' is a gpu-accelerated implementation of the Fast Phase Retrieval algorithm for solving the support-constrained Phase Retrieval problem on near-Schwarz objects. 

Near-Schwarz objects are defined by the phase of their Z-Transform:
    $$ |\text{arg}(X(\textbf{z})) - \text{arg}(\textbf{z}^\textbf{n})| \leq \frac{\pi}{2} $$

Fast Phase Retrieval is guaranteed to work for objects known as "near-Schwarz Objects": objects with a Z-Transform
## Installation

Install the stable version of `phastphase` from [PyPI](https://pypi.org/project/phastphase/) using:

```console
pip install phastphase
```

Install the latest version of `phastphase` from GitHub using:

```console
pip install git+https://github.com/slmsuite/phastphase
```