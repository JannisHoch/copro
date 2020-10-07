Model execution
=========================

To be able to run the model, the conda environment has to be activated first.

.. code-block:: console

    $ conda activate copro

Example notebook
-----------------

There are jupyter notebooks available to guide you through the model application process (also see :ref:`workflow`).
They can all be run and converted to htmls by executing the provided shell-script.

.. code-block:: console

    $ cd path/to/copro/example
    $ sh run.sh

It is of course also possible to execute the notebook cell by cell using jupyter notebook.

Runner script
----------------

To run the model from command line, a command line script is provided. 
All data and settings are retrieved from the settings-file which needs to be provided as inline argument.

.. code-block:: console

    $ cd path/to/copro/scripts
    $ python runner.py ../example/example_settings.cfg

By default, output is stored to the output directory specified in the settings-file. 