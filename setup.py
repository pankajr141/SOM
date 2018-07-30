'''
Created on Jul 30, 2018

@author: 703188429
'''
import setuptools
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="somlib",
    version="0.0.4",
    author="Pankaj Rawat",
    author_email="pankajr141@gmail.com",
    description="This is python implementation for Kohonen Self Organizing map using numpy and tensor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pankajr141/SOM",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
          'numpy',
          'tensorflow',
          'pandas',
          'scipy',
          'matplotlib',
    ],
)