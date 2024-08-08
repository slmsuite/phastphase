*************
API Reference
*************

This page provides an auto-generated summary of the internals of |fastphase|_'s API
outside of the :meth:`~fastphase.retrieve()` function.
The general user should not need to look at these internals.
You can find the source on `GitHub <https://github.com/QPG-MIT/fastphase>`_.

Self-contained algorithms used for steps of retrieval are partially separated into their
own files.

.. currentmodule:: fastphase.algorithms
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   conjgradsolve
   supernewton

Alongside definitions of loss functions and other helper functions.

.. currentmodule:: fastphase.algorithms
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   loss
   helper

.. |fastphase| replace:: :mod:`fastphase`
.. _fastphase: https://github.com/QPG-MIT/fastphase