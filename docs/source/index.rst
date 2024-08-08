fastphase
=========

|fastphase|_ solves the phase retrieval problem:

   Given a 2D farfield amplitude :math:`|y|`, what is the best fit complex 2D nearfield :math:`x`?

   .. math::
      x = \text{retreive}\left(|y|\right), \,\,\, \text{ where } y \equiv \mathcal{F}(x).

This functionality is available under :meth:`fastphase.retrieve_phase()`. TODO

.. autofunction:: fastphase.retrieve_phase

.. toctree::
   :maxdepth: 2
   :caption: TODO

   installation
   examples
   api

.. |fastphase| replace:: :mod:`fastphase`
.. _fastphase: https://github.com/slmsuite/fastphase