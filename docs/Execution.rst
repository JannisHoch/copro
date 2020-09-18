Model execution
=========================

To be able to run the model, the conda environment has to be activated first.

.. code-block:: console

    $ conda activate conflict_model

Example notebook
-----------------

To run the example jupyter notebook, follow these instructions

.. code-block:: console

    $ cd path/to/conflict_model/example
    $ sh run.sh

This automatically executes the notebook and converts it to a html-file, also stored in the example folder.

It is of course also possible to execute the notebook cell by cell using jupyter notebook.

Runner script
----------------

To run the model from command line, a command line script is provided. 
All data and settings are retrieved from the settings-file which needs to be provided as inline argument.

.. code-block:: console

    $ cd path/to/conflict_model/scripts
    $ python runner.py path/to/conflict_model/data/run_setting.cfg

By default, output is stored to the output directory specified in the settings-file. 