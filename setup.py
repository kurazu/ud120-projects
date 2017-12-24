#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ud120-projects',
    version='0.1',
    description='Machine Learning Intro',
    author='Tomasz Maćkowiak',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'scipy',
        'sklearn',
        'matplotlib',
        'nltk'
    ]
)
