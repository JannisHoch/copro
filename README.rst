===============
Overview
===============

The conflict_model
----------------
(Machine learning) model for mapping environmental drivers of conflict risk.

.. image:: https://travis-ci.com/JannisHoch/conflict_model.svg?token=BnX1oxxHRbyd1dPyXAp2&branch=dev
    :target: https://travis-ci.com/JannisHoch/conflict_model

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://github.com/JannisHoch/conflict_model/blob/dev/LICENSE

.. image:: https://badges.frapsoft.com/os/v2/open-source.svg?v=103
    :target: https://github.com/ellerbrock/open-source-badges/

Installation
----------------

To install the conflict model, first clone the code from GitHub. It is advised to create an individual python environment first. 
You can then install the model package into this environment.

.. code-block:: console

    $ git clone https://github.com/JannisHoch/conflict_model.git
    $ cd path/to/conflict_model
    $ conda env create -f environment.yml
    $ conda activate conflict_model
    $ python setup.py develop

Execution
----------------

Example notebook
^^^^^^^^^^^^^^^^^^

To run the example jupyter notebook, follow these instructions

.. code-block:: console

    $ cd path/to/conflict_model/example
    $ sh run.sh

This automatically executes the notebook and converts it to a html-file, also stored in the example folder.

It is of course also possible to execute the notebook cell by cell using jupyter notebook.

Runner script
^^^^^^^^^^^^^^^^^^

To run the model from command line, a command line script is provided. 
All data and settings are retrieved from the settings-file which needs to be provided as inline argument.

.. code-block:: console

    $ cd path/to/conflict_model/scripts
    $ python runner.py path/to/conflict_model/data/run_setting.cfg

By default, output is stored to the output directory specified in the settings-file. 

Authors
----------------

* Jannis M. Hoch (Utrecht University)
* Sophie de Bruin (Utrecht University, PBL)
* Niko Wanders (Utrecht University)

Corrosponding author: Jannis M. Hoch (j.m.hoch@uu.nl)