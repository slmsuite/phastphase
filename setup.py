"""
setup.py - this module makes the package installable
"""

from setuptools import setup

NAME = "phasefast"
VERSION = "0.0.1"
DEPENDENCIES = [
    "numpy",
    "torch",
    "torchaudio",
    "torchvision",
    "pytorch-minimize",
]
DESCRIPTION = (
    "Accurate solution to the phase retrieval problem "
    "for near-Schwarz objects."
)
AUTHOR = "phasefast developers"
AUTHOR_EMAIL = "cbrabec@mit.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
)
