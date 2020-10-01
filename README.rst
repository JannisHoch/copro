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

.. image:: https://readthedocs.org/projects/conflict-model/badge/?version=dev
    :target: https://conflict-model.readthedocs.io/en/dev/?badge=dev

.. image:: https://img.shields.io/github/v/release/JannisHoch/conflict_model
    :target: https://github.com/JannisHoch/conflict_model/releases/tag/v0.0.5-pre
    
.. image:: https://zenodo.org/badge/254407279.svg
   :target: https://zenodo.org/badge/latestdoi/254407279

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

To be able to run the model, the conda environment has to be activated first.

.. code-block:: console

    $ conda activate conflict_model

Example notebook
^^^^^^^^^^^^^^^^^^

There are jupyter notebooks available to guide you through the model application process.
They can all be run and converted to htmls by executing the provided shell-script.

.. code-block:: console

    $ cd path/to/conflict_model/example
    $ sh run.sh

It is of course also possible to execute the notebook cell by cell using jupyter notebook.

Runner script
^^^^^^^^^^^^^^^^^^

To run the model from command line, a command line script is provided. 
All data and settings are retrieved from the settings-file which needs to be provided as inline argument.

.. code-block:: console

    $ cd path/to/conflict_model/scripts
    $ python runner.py ../example/example_settings.cfg

By default, output is stored to the output directory specified in the settings-file. 

Documentation
---------------

Model documentation including model API can be found at http://conflict-model.rtfd.io/

Code of conduct and Contributing
---------------------------------

Please find the relevant information on our Code of Conduct and how to contribute to this package in the relevant files.

Authors
----------------

* Jannis M. Hoch (Utrecht University)
* Sophie de Bruin (Utrecht University, PBL)
* Niko Wanders (Utrecht University)

Corrosponding author: Jannis M. Hoch (j.m.hoch@uu.nl)
