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

This help information can be also accessed with

.. code-block:: console

    $ copro_runner --help

All data and settings are retrieved from the configuration-file (cfg-file) which needs to be provided as command line argument.
A typical command would thus look like this:

.. code-block:: console

    $ copro_runner settings.cfg

In the cfg-file, the various settings of the simulation are defined.
For more details, see INSERT LINK.

In case issues occur, updating ``setuptools`` may be required.

.. code-block:: console

    $ pip3 install --upgrade pip setuptools

Reference run
^^^^^^^^^^^^^^^^

In the reference run, the sample data (X) and target data (Y) are read and stored in arrays along with their geographic information.
A scaling technique is used to normalize the sample data as the range, magnitude, and units of the sample data can vary between input files.
This data is then split into a training and test set. While the former is used to fit the model, the latter is used to evaluate the accuracy of a prediction made with this fitted model.
To increase the robustness of the split-sample test, this step can be repeated multiple times to obtain an averaged picture of model accuracy.
After each repetition, the model outcome is associated to its geographic origin to yield maps of conflict risk.

At the end of the reference run, the classifier is fitted on more time with all sample and target data. It is then stored to be used in one (or more) projection runs.

Projection runs
^^^^^^^^^^^^^^^^

The projections runs employ the fitted classifier of the reference run in conjunction with other sample data, for example for future scenarios. 
Based on the relations established between sample data and target data of the reference run, the model projects where conflict will occur.

.. important:: 

    In order to re-use the classifier, the number of sample data features used in the projection runs must be identical to the feature number used in the reference run.

Binder
--------

There is also a notebook running on `Binder <https://mybinder.org/v2/gh/JannisHoch/copro/update_docs?filepath=%2Fexample%2Fnb_binder.ipynb>`_. 

Please check it out to go through the model execution step-by-step and interactively explore the functionalities of CoPro.