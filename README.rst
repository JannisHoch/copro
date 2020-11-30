===============
Overview
===============

CoPro
----------------

Welcome to CoPro, a machine-learning tool for conflict risk projections based on climate, environmental, and societal drivers.

.. image:: https://travis-ci.com/JannisHoch/copro.svg?branch=dev
    :target: https://travis-ci.com/JannisHoch/copro

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://github.com/JannisHoch/copro/blob/dev/LICENSE

.. image:: https://readthedocs.org/projects/copro/badge/?version=latest
    :target: https://copro.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/github/v/release/JannisHoch/copro
    :target: https://github.com/JannisHoch/copro/releases/tag/v0.0.6

.. image:: https://zenodo.org/badge/254407279.svg
    :target: https://zenodo.org/badge/latestdoi/254407279

.. image:: https://badges.frapsoft.com/os/v2/open-source.svg?v=103
    :target: https://github.com/ellerbrock/open-source-badges/

Model purpose
--------------

CoPro employes observed conflict data together with (user-provided) socio-economic and environmental data to train different classifiers (RFClassifier, kNearestClassifier, and Support Vector Classifier).
As primary model output, conflict risk (defined as the fraction conflict predictions of all predictions) is provided.
To capture the geographical variability of conflict and socio-environmental drivers, the model is spatially explicit and calculates conflict risk at a (user-specified) aggregation level.
This way, the model is able to also capture the relevant sub-national variability of conflict and conflict drivers.

In addition to the calculation of conflict risk, can the model, for instance, be used to make scenario projections, evaluate the relative feature importances, or benchmark different datasets.

All in all, CoPro supports the mapping of current and future areas at risk of conflict, while also facilitating obtaining a better understanding of the underlying processes.

Installation
----------------

To install copro, first clone the code from GitHub. It is advised to create an individual python environment first. 
You can then install the model package into this environment.

.. code-block:: console

    git clone https://github.com/JannisHoch/copro.git
    cd path/to/copro
    conda env create -f environment.yml
    conda activate copro

To install CoPro in editable mode in this environment, run this command next:

.. code-block:: console

    pip install -e <path/to/copro>/copro

Command-line script
--------------------

To be able to run the model, the conda environment has to be activated first.

.. code-block:: console

    conda activate copro

To run the model from command line, a command line script is provided. The usage of the script is as follows:

.. code-block:: console

    Usage: runner.py [OPTIONS] CFG

    Main command line script to execute the model.  All settings are read from
    cfg-file. One cfg-file is required argument to train, test, and evaluate
    the model. Additional cfg-files can be provided as optional arguments,
    whereby each file corresponds to one projection to be made.

    Args:     CFG (str): (relative) path to cfg-file

    Options:
    -proj, --projection-settings PATH   path to cfg-file with settings for a projection run

    -v, --verbose                       command line switch to turn on verbose mode
    --help                              Show this message and exit.

This help information can be also accessed with

.. code-block:: console

    python copro_runner.py --help

All data and settings are retrieved from the settings-file (cfg-file) which needs to be provided as inline argument.

Example data
----------------

Example data for demonstration purposes can be downloaded from `Zenodo <https://zenodo.org/record/4297295>`_.

With this (or other) data, the provided configuration-files (cfg-files) can be used to perform a reference run or a projection run. 
All output is stored in the output directory specified in the cfg-files. 

Jupyter notebooks
^^^^^^^^^^^^^^^^^^

There are multiple jupyter notebooks available to guide you through the model application process step-by-step.
They can all be run and converted to htmls by executing the provided shell-script.

.. code-block:: console

    cd path/to/copro/example
    sh run.sh

It is of course also possible to execute the notebook cell-by-cell and explore the full range of possibilities.

Command-line
^^^^^^^^^^^^^^^^^^

While the notebooks are great for exploring, the command line script is the envisaged way to use CoPro.

To only test the model for the reference situation, the cfg-file is the required argument.

To make a projection, both cfg-files need to be specified with the latter requiring the -proj flag.
If more projections are ought to be made, multiple cfg-files can be provided with the -proj flag.

.. code-block:: console

    cd path/to/copro/scripts
    python copro_runner.py ../example/example_settings.cfg
    python copro_runner.py ../example/example_settings.cfg -proj ../example/example_settings_proj.cfg

Validation
^^^^^^^^^^^^^^^^^^

The reference model makes use of the `UCDP Georeferenced Event Dataset <https://ucdp.uu.se/downloads/index.html#ged_global>`_ for observed conflict. 
The selected classifier is trained and validated against this data.

Main validation metrics are the ROC-AUC score as well as accuracy, precision, and recall. 
All metrics are reported and written to file per model evaluation.

With the example data downloadable from `Zenodo <https://zenodo.org/record/4297295>`_, a ROC-AUC score of 0.82 can be obtained. 
Note that with additional and more explanatory sample data, the score will most likely increase.

.. figure:: docs/_static/roc_curve.png

Documentation
---------------

Extensive model documentation including full model API description can be found at http://copro.rtfd.io/

Code of conduct and Contributing
---------------------------------

The project welcomes contributions from everyone! 
To make collaborations as pleasant as possible, we expect contributors to the project to abide by the Code of Conduct.

License
--------

CoPro is released under the MIT license.

Authors
----------------

* Jannis M. Hoch (Utrecht University)
* Sophie de Bruin (Utrecht University, PBL)
* Niko Wanders (Utrecht University)

Corresponding author: Jannis M. Hoch (j.m.hoch@uu.nl)
