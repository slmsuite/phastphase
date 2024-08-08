*********
Retrieval
*********

|fastphase|_ solves the phase retrieval problem:

   Given a 2D farfield amplitude :math:`|y|`, what is the complex 2D nearfield :math:`x`?

   .. math::
      x = \text{retreive}\left(|y|\right), \,\,\, \text{ where } y \equiv \mathcal{F}(x).

This functionality is available under :meth:`fastphase.retrieve()`. TODO


.. autofunction:: fastphase.retrieve


.. |fastphase| replace:: :mod:`fastphase`
.. _fastphase: https://github.com/QPG-MIT/fastphase