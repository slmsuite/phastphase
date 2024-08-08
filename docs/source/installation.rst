.. _installation:

Installation
============

.. PyPi
.. ----

.. Install the stable version of |fastphase|_ from `PyPi <https://pypi.org/project/fastphase/>`_ using:

.. .. code-block:: console

..     pip install fastphase

GitHub
------

Install the latest version of |fastphase|_ from `GitHub <https://github.com/slmsuite/fastphase>`_ using:

.. code-block:: console

    pip install git+https://github.com/slmsuite/fastphase

One can also clone |fastphase|_ directly and add its directory to the Python path.
*Remember to install the dependencies (next sections)*.

.. code-block:: console

    git clone https://github.com/slmsuite/fastphase

Required Dependencies
---------------------

The following python packages are necessary to run |fastphase|_. These are listed as PyPi
dependencies and thus are installed automatically if PyPi (``pip``) is used to install.

- `python <https://www.python.org/>`_
- `numpy <https://numpy.org/>`_
- `torch <https://scipy.org/>`_
- `torchaudio <https://scipy.org/>`_
- `torchvision <https://scipy.org/>`_
- `pytorch-minimize <https://pytorch-minimize.readthedocs.io/en/latest/>`_

One can also install these dependencies directly.

.. code-block:: console

    pip install -r requirements.txt

.. |fastphase| replace:: :mod:`fastphase`
.. _fastphase: https://github.com/slmsuite/fastphase