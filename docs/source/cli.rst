Command line script
===================

`copro` contains a command line script which is automatically compiled when following the :ref:`Installation` instructions.
It runs the evaluation stage: it fits and tests the model for the reference period.
Projections are made from the deploy model, see :ref:`Scripts for deployment`.

Information about the script is shown with the following command:

.. code-block:: bash

    copro_runner --help

This should yield the following output:

.. code-block::

    Usage: copro_runner [OPTIONS] CFG

      Main command line script to execute the model.

      Args:     CFG (str): (relative) path to cfg-file

    Options:
      -c, --cores INTEGER    Number of jobs to run in parallel. Default is 5.
      -v, --verbose INTEGER  Verbosity level of the output. Default is 0.
      --help                 Show this message and exit.

Output
------

In ``<output_dir>/<simulation_name>/_REF``:

* ``XY.gpkg``: the feature matrix with polygon geometry
* ``estimators/``: the fitted estimator of each run
* ``evaluation_metrics.csv``: scores per run
* ``perm_importances.parquet``: permutation importance on the test rows, one row per run and repeat
* ``raw_output_data.npy``: test-row predictions of all runs
* ``output_for_REF.gpkg``: model accuracy per polygon
* ``files/conflicts_in_<y_end>.csv`` and ``files/last_conflict_year_<y_end>.csv``: conflict state at the end of the reference period

In ``<output_dir>/eval``:

* ``<simulation_name>_<split_strategy>.csv``: one row per held-out row and run, with ``run_nr``, ``ID``, ``year``, ``conflict_t_min_1``, ``y_test``, ``y_pred`` and ``y_prob_1``
* ``<simulation_name>_<split_strategy>.json``: what produced these rows - predictor set, period, scaler, hyperparameters and feature columns
