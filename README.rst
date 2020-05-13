===============
Overview
===============

conflict_model
----------------
(Machine learning) model for mapping environmental drivers of conflict risk

installation
----------------

to install the conflict model, first clone the code from GitHub. It is advised to create an individual python environment first. Then go to the model folder and install the model.

.. code-block:: console

    $ git clone https://github.com/JannisHoch/conflict_model.git
    $ cd path/to/conflict_model
    $ conda-env create -f=environment.yml
    $ python setup.py develop

execution
----------------

example notebook
^^^^^^^^^^^^^^^^^^

To run the example jupyter notebook, follow these instructions

.. code-block:: console

    $ cd path/to/conflict_model/example
    $ sh run.sh

This automatically executes the ipynb and converts it to a html-file, also stored in the conflict_model folder.

with runner script
^^^^^^^^^^^^^^^^^^

To run the model from command line, a command line script is provided.

.. code-block:: console

    $ cd path/to/conflict_model/scripts
    $ python runner.py path/to/conflict_model/data/run_setting.cfg

For help, try this if you are in the scripts folder:

.. code-block:: console

    $ python runner.py --help

authors
----------------
Jannis M. Hoch (Utrecht University), Sophie de Bruin (Utrecht University, PBL), Niko Wanders (Utrecht University)

corrosponding author: Jannis M. Hoch (j.m.hoch@uu.nl)

license
----------------
tba