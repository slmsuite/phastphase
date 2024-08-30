.. _installation:

Installation
============

PyPI
----

Install the stable version of |phastphase|_ from `PyPI <https://pypi.org/project/phastphase/>`_ using:

.. code-block:: console

    pip install phastphase

GitHub
------

Install the latest version of |phastphase|_ from `GitHub <https://github.com/slmsuite/phastphase>`_ using:

.. code-block:: console

    pip install git+https://github.com/slmsuite/phastphase

One can also clone |phastphase|_ directly and add its directory to the Python path.
*Remember to install the dependencies (next sections)*.

.. code-block:: console

    git clone https://github.com/slmsuite/phastphase

Required Dependencies
---------------------

The following python packages are necessary to run |phastphase|_. These are listed as PyPI
dependencies and thus are installed automatically if PyPI (``pip``) is used to install.

- `python <https://www.python.org/>`_
- `numpy <https://numpy.org/>`_
- `torch <https://scipy.org/>`_
- `torchaudio <https://scipy.org/>`_
- `torchvision <https://scipy.org/>`_
- `pytorch-minimize <https://pytorch-minimize.readthedocs.io/en/latest/>`_

One can also install these dependencies directly.

.. code-block:: console

    pip install -r requirements.txt

.. |phastphase| replace:: :mod:`phastphase`
.. _phastphase: https://github.com/slmsuite/phastphase