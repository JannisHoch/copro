#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = [
    "click==8.1.7",
    "xarray==2023.12.0",
    "pandas==2.1.4",
    "rasterio==1.2.10",
    "rioxarray==0.15.0",
    "rasterstats==0.19.0",
    "geopandas==0.14.2",
    "numpy==1.26.3",
    "scikit-learn==1.5.0",
    "netcdf4==1.6.0",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Jannis M. Hoch",
    author_email="j.m.hoch@uu.nl",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Python-model build on scikit-learn functions, designed to facilitate the set-up, execution, and evaluation of machine-learning models for the study of the climate-conflict nexus.",
    entry_points={
        "console_scripts": [
            "copro_runner=copro.scripts.copro_runner:cli",
        ],
    },
    install_requires=requirements,
    license="MIT",
    long_description=readme,
    include_package_data=True,
    keywords="conflict, climate, machine learning, projections",
    name="copro",
    packages=find_packages(
        include=["copro", "copro.*"], exclude=["docs", "tests", "joss"]
    ),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://copro.readthedocs.io/",
    version="2.0.1",
    zip_safe=False,
)
