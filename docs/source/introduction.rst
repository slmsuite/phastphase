
Introduction
==========

Support-Constrainted Phase Retrieval
------------------------------------
``phastphase`` is a package for solving the support-constrained phase retrieval problem: Given a set of intensities :math:`\textbf{y}`, find a
complex object :math:`\textbf{x}` satisfying:

:math:`\left| \mathcal{F}\{\textbf{x}\} \right|^2 \approx \textbf{y}`

Where :math:`\mathcal{F}` is the zero padded DFT - i.e. :math:`\textbf{x}` is zero-padded to the size of :math:`\textbf{y}`

For the problem to be well-posed, the length of :math:`\textbf{y}` must be at least twice the length of :math:`\textbf{x}` in each dimension.
Formally if :math:`\textbf{m}` is the shape-tuple of :math:`\textbf{y}`, and :math:`\textbf{n}` 
is the shape-tuple of :math:`\textbf{x}` we must have: :math:`\textbf{m} \geq 2 \textbf{n}`



phastphase and Schwarz Objects
------------------------------
Phase Retrieval is strongly NP-Hard, so only specific instances can be solved efficiently. ``phastphase``  works for a class of objects known
as Schwarz objects. These are objects defined by the phase of their Z-transforms. To be a Schwarz object, on the poly-Torus there must exist
and integer tuple :math:`\textbf{w}` such that the Z-transform satisfies.

:math:`|\text{arg}(X(\textbf{z})) - \text{arg}(\textbf{z}^\textbf{w})| \leq \frac{\pi}{2}`

:math:`\textbf{w}` is known as the "winding tuple" of :math:`\textbf{x}` and is automatically determined by ``phastphase`` 
from :math:`\textbf{y}`


Why ``phastphase``?
-------------------
``phastphase`` implements the Fast Phase Retrieval algorithm, the first polynomial time algorithm to provably succeed at Fourier phase retrieval 
without requiring phase masks or nearfield information. Other features include:

* It's accurate! Existing algorithms such as Wirtinger flow fail on Fourier phase retrieval without
  gaussian random phase masks. Even with the masks, existing methods can fail on many cases.  

* It's phast! Using GPU acceleration recoveries take seconds on full-HD images. The underlying algorithm
  has a worst case arithmetic complexity of :math:`O(N\log(N)` for images with N pixels.

* ``phastphase`` can use nearfield information if available to broaden the class of objects it can recover.

* User Friendly! ``phastphase`` takes both numpy arrays and pytorch tensors!



Basic Use
----------
Phase retrieval can be performed by invoking :meth:`phastphase.retrieve()`. There is only one required argument, the
intensities :math:`\textbf{y}`. However, also supplying the shape tuple of :math:`\textbf{x}` is **highly** encouraged. If the shape 
tuple of :math:`\textbf{x}` is not supplied, the method will assume critical oversampling. The critical sampling assumption
appears to work well most of the time, but leads to poor behavior in some cases.


