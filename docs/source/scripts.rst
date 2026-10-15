Scripts for deployment
======================

Four scripts take a model from evaluation to projection.
They are run with ``python -m``, in this order.
None of them deletes anything.

Fit the deploy model
--------------------

.. code-block:: bash

    python -m copro.scripts.deploy <config.yaml> --out-dir <deploy_dir> --cores 8

Fits one estimator on all rows, with the scaler fitted on the same rows, and writes them to
``<deploy_dir>/_DEPLOY`` together with ``deploy_manifest.json``.
The manifest records the predictor set, the feature columns in order, the polygons covered,
the row counts and the hyperparameters as fitted.
Hyperparameters, and whether a search runs, come from the configuration exactly as in the evaluation run.

``general.feature_set`` is required.
The configuration file and its projection files are copied into ``_DEPLOY``, so the projection
does not depend on files that may have changed since.

Fit the score-to-probability rules (optional)
---------------------------------------------

.. code-block:: bash

    python -m copro.scripts.calibrate <deploy_dir> --rows <eval csv> [--check-rows <eval csv>]

The model output ``y_prob_1`` is a score: it ranks polygon-years, but its level is not a probability.
This script measures, on the held-out rows of an evaluation run, how often conflict followed a given score,
and writes one rule per stratum (no conflict in the previous year, and conflict in the previous year)
to ``_DEPLOY/calibration.json``.
Rows are grouped along the score in groups of at least ``--min-rows`` rows (default 100),
and the group frequencies are made monotone with isotonic regression.

``--rows`` should come from a ``leave_polygons_out`` run, whose models saw every year, as the deploy model does.
``--check-rows`` applies the rules to a second set of rows, for example from a ``temporal`` run,
and writes expected against observed conflict to ``calibration_check.csv``.

The rows must have been produced with the deploy model's configuration.
The script compares the json written beside the rows with the deploy configuration and stops if they differ.

Project
-------

.. code-block:: bash

    python -m copro.scripts.project <deploy_dir> [--keys <key> ...]

Runs the projections listed in the deploy configuration, or only the given keys,
and writes them to ``<deploy_dir>/_PROJ/<key>``.
Only the polygons covered by the deploy model are scored.
A missing input value for a covered polygon stops the run, naming the features and polygons affected.
If ``calibration.json`` is present, the output carries a ``calibrated_probability`` beside the score.

Per year, the projection writes ``output_in_<year>.gpkg``, and per estimator
``projection_for_<year>.csv`` and ``X_for_<year>.csv``, the matrix exactly as scored.

Explain (optional)
------------------

.. code-block:: bash

    python -m copro.scripts.explain <deploy_dir> --key <key> --year <year>

Splits each polygon's score into one contribution per feature with TreeSHAP and writes
``<deploy_dir>/_PROJ/<key>/shap_in_<year>_seed<seed>.csv``.
The baseline is a sample of the region's polygon-years without conflict in the previous year,
taken from the rows the deploy model was fitted on, so a contribution says how much a feature
lifts a polygon above a typical peaceful polygon in its region.

This script needs ``shap``, an optional dependency (``pip install copro[explain]``),
and the XY cache named by ``pre_calc.XY``.
