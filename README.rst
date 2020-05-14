===============
Overview
===============

conflict_model
----------------
(Machine learning) model for mapping environmental drivers of conflict risk

installation
----------------

To install the conflict model, first clone the code from GitHub. It is advised to create an individual python environment first. Then go to the model folder and install the model.

.. code-block:: console

    $ git clone https://github.com/JannisHoch/conflict_model.git
    $ cd path/to/conflict_model
    $ conda-env create -f=environment.yml
    $ conda activate conflict_model
    $ python setup.py develop

execution
----------------

example notebook
^^^^^^^^^^^^^^^^^^

To run the example jupyter notebook, follow these instructions

.. code-block:: console

    $ cd path/to/conflict_model/example
    $ sh run.sh

This automatically executes the notebook and converts it to a html-file, also stored in the example folder.

with runner script
^^^^^^^^^^^^^^^^^^

To run the model from command line, a command line script is provided. In the most basic version, all data is taken from the settings-file.

.. code-block:: console

    $ cd path/to/conflict_model/scripts
    $ python runner.py path/to/conflict_model/data/run_setting.cfg

.. note:: by default, no output is stored in the current version of the model!

If output is to be stored in an output map, this currently needs to be specified in the runner scipt explictely (-s option).
By default, output is stored to the output directory specified in the settings-file. Alternatively, this can be provided via command line too (-o option)

.. code-block:: console

    $ python runner.py -s True -o path/to/output/folder path/to/conflict_model/data/run_setting.cfg

For help, try this if you are in the scripts folder:

.. code-block:: console

    $ python runner.py --help

authors
----------------

* Jannis M. Hoch (Utrecht University)
* Sophie de Bruin (Utrecht University, PBL)
* Niko Wanders (Utrecht University)

corrosponding author: Jannis M. Hoch (j.m.hoch@uu.nl)

license
----------------
tba