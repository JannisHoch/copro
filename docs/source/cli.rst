Command line script
===================

`copro` contains a command line script which is automatically compiled when following the :ref:`Installation` instructions.

Information about the script can be run with the following command:

.. code-block:: bash

    copro_runner -help

This should yield the following output:

.. code-block::

    Usage: copro_runner [OPTIONS] CFG

    Main command line script to execute the model.

    Args:     CFG (str): (relative) path to cfg-file

    Options:
    -c, --cores INTEGER    Number of jobs to run in parallel. Default is 0.
    -v, --verbose INTEGER  Verbosity level of the output. Default is 0.
    --help                 Show this message and exit.