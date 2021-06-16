Model execution
=========================

To be able to run the model, the conda environment has to be activated first.

.. code-block:: console

    $ conda activate copro

.. _script:

Runner script
----------------

To run the model, a command line script is provided. The usage of the script is as follows:

.. code-block:: console

    Usage: copro_runner [OPTIONS] CFG

    Main command line script to execute the model. 
    All settings are read from cfg-file.
    One cfg-file is required argument to train, test, and evaluate the model.
    Multiple classifiers are trained based on different train-test data combinations.
    Additional cfg-files for multiple projections can be provided as optional arguments, whereby each file corresponds to one projection to be made.
    Per projection, each classifiers is used to create separate projection outcomes per time step (year).
    All outcomes are combined after each time step to obtain the common projection outcome.

    Args:     CFG (str): (relative) path to cfg-file

    Options:
    -plt, --make_plots        add additional output plots
    -v, --verbose             command line switch to turn on verbose mode

Help information can be accessed with

.. code-block:: console

    $ copro_runner --help

All data and settings are retrieved from the configuration-file (``cfg-file``, see :ref:`Settings` ) which needs to be provided as command line argument.
In the cfg-file, the various settings of the simulation are defined.

A typical command would thus look like this:

.. code-block:: console

    $ copro_runner settings.cfg

In case issues occur, updating ``setuptools`` may be required.

.. code-block:: console

    $ pip3 install --upgrade pip setuptools

Binder
--------

There is also a notebook running on `Binder <https://mybinder.org/v2/gh/JannisHoch/copro/dev?filepath=%2Fexample%2Fnb_binder.ipynb>`_. 

Please check it out to go through the model execution step-by-step and interactively explore the functionalities of CoPro.
