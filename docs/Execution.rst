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
The notebooks should give a fairly good impression how the model can be executed function-by-function and what functionality these functions have.

.. _script:

Runner script
----------------

To run the model from command line, a command line script is provided. 
The number of inline arguments differs whether only a reference run or also one or more projections runs are executed.

By default, output is stored to the output directory specified in the individual configurations-file (cfg-file). 

Reference run
^^^^^^^^^^^^^^^^
All data and settings are retrieved from the cfg-file (see :ref:`settings`).
Based on these settings, data is sampled and the model is trained, tested, and evaluated.
The output is then stored to the output directory.

.. code-block:: console

    $ cd path/to/copro/scripts
    $ python runner.py ../example/example_settings.cfg

Projection runs
^^^^^^^^^^^^^^^^
If also projections are computed, multiple additional cfg-files can be provided.
For each projection, one individual cfg-file is required.

Since the projections are based on the reference run, at least two cfg-file are needed.
The command would then look like this:

.. code-block:: console

    $ cd path/to/copro/scripts
    $ python runner.py ../example/example_settings.cfg -proj ../example/example_settings_proj.cfg

.. info::

    Multiple projections can be made by specifying various cfg-files with the -proj flag.

Help
^^^^^^^^^^^^^^^^
For further help how to use the script, try this:

.. code-block:: console

    $ python runner.py --help
