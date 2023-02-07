from setuptools import setup
from setuptools import find_packages


setup(name='ppo',
      version='0.0.1',
      extras_require={
          'gym': ['gym'],
      },
      install_requires=["tensorflow", "numpy", "keras"],
      packages=find_packages())