#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['Click>=7.0', 
                'xarray==0.15.1',
                'pandas==1.0.3',
                'rasterio==1.1.0',
                'rioxarray>=0.0.26',
                'rasterstats==0.14',
                'geopandas==0.8.0',
                'numpy==1.18.1',
                'scikit-learn>=0.22.1',]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Jannis M. Hoch",
    author_email='j.m.hoch@uu.nl',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="modelling interplay between conflict and climate",
    entry_points={},
    install_requires=requirements,
    license="MIT",
    long_description='blablabla',
    include_package_data=True,
    keywords='conflict, climate, machine learning, projections',
    name='copro',
    packages=find_packages(include=['copro', 'copro.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JannisHoch/copro',
    version='0.0.5',
    zip_safe=False,
)
