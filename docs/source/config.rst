Model configuration file
========================

The command line script and the scripts in :ref:`Scripts for deployment` take a
reference configuration file in YAML format.
It holds everything needed to fit and evaluate the model over the reference period.

Each projection is described in a separate, shorter configuration file.
The reference file lists them in its ``projections`` block, one entry per projection key.

.. note::
    Files named in ``data`` are relative to ``general.input_dir``.
    A relative ``input_dir`` is itself relative to the folder holding the configuration file.

Reference configuration file
----------------------------

.. code-block:: yaml

    general:
      # label for the predictor set; recorded with the evaluation rows and
      # in the deploy manifest. Required by deploy.py
      feature_set: core
      input_dir: ./example_data
      output_dir: ./OUT
      simulation_name: test_run
      # first and last year of the reference period
      y_start: 2000
      y_end: 2012
      # optional: worker count for the zonal statistics, default -1 (all cores)
      n_jobs_zonal: -1

    projections:
      proj_nr_1:
        file: example_settings_proj.yaml
        # last year of this projection
        y_end: 2015

    pre_calc:
      # path to a pickled XY matrix; "" switches the cache off
      XY: ""

    data:
      extent:
        file: path/to/polygons.gpkg
        # optional: polygon ID column, default watprovID
        id: watprovID

      conflict:
        file: path/to/conflict.csv
        thresholds:
          fatalities:
            vmin: 5
          # 1=state-based armed conflict; 2=non-state conflict; 3=one-sided violence
          type_of_violence:
            values: [1, 2, 3]

      indicators:
        # the key must be identical to the variable name in the nc-file
        precipitation:
          file: path/to/precipitation_data.nc
          log: False
          stat: mean
        temperature:
          file: path/to/temperature_data.nc
          log: False
          stat: min

    machine_learning:
      model: Classification
      # choose from: MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
      scaler: QuantileTransformer
      train_fraction: 0.7
      # number of model instances
      n_runs: 3
      # random, temporal or leave_polygons_out
      split_strategy: leave_polygons_out
      split_seed: 42
      # required for split_strategy temporal: train on years up to and
      # including this one, test on later years
      temporal_cutoff: 2008
      # off when not given
      tune_hyperparameters: False
      # optional: replaces the package default for the keys given
      hyperparameters:
        class_weight:
          1: 100

general
^^^^^^^

``simulation_name`` names the run folder ``<output_dir>/<simulation_name>``.
That folder is deleted and recreated at the start of every run.
Per-row evaluation results go to ``<output_dir>/eval`` instead, which is not cleared,
so results from different split strategies accumulate side by side.

projections
^^^^^^^^^^^

``file`` is relative to the folder holding the reference configuration file.
Keep the projection files in that same folder: ``deploy.py`` copies each one into
the deploy folder under its file name, and ``project.py`` looks for it there.

pre_calc
^^^^^^^^

``XY`` caches the feature matrix, which avoids reading every NetCDF file again.
If the file exists, it is loaded and checked against the column list, ``y_start``,
``y_end``, the polygon file and the conflict file, and the run stops on any mismatch.
If the path is given but the file does not exist yet, the matrix is computed and written there.

.. important::
    The check compares names and years, not file contents.
    Delete the cache by hand after changing an input file in place,
    or after changing how conflict features are computed.

``scripts/explain.py`` needs this cache: it takes its background rows from it.

data
^^^^

The polygon file must have a ``watprovID`` column: several functions use that name directly.

The conflict file is a csv with ``year``, ``latitude``, ``longitude`` and ``event_id_cnty`` columns.
``thresholds`` filters it per column, with ``values`` (keep these values),
``vmin`` (keep values at or above) or ``vmax`` (keep values at or below).
The version 1 keys ``min_nr_casualties`` and ``type_of_violence`` are still accepted
and translated to ``fatalities: {vmin: ...}`` and ``type_of_violence: {values: ...}``, with a warning.
If a column appears both in ``thresholds`` and as a version 1 key, ``thresholds`` is used.

For indicators, the value of year *t - 1* is used for year *t*.
``stat`` is any statistic supported by rasterstats, ``mean`` by default.
``log: True`` is not implemented and stops the run.

Rows with any missing value are dropped before fitting, and the run reports
which indicators and polygons were affected.
A polygon that loses all its rows is absent from the evaluation and is not projected.

machine_learning
^^^^^^^^^^^^^^^^

``split_strategy`` decides how each run separates training and test rows:

* ``random``: a random share of rows is held out. The same polygon can appear in both sets, so results are optimistic.
* ``temporal``: years after ``temporal_cutoff`` are held out.
* ``leave_polygons_out``: whole polygons are held out, so no polygon appears in both sets.

The scaler is fitted on the training rows of each run only.
The seed of run *n* is ``split_seed + n``, so repeated runs draw different splits and the result is reproducible.
With ``temporal`` and the search off, every run would be identical, so the model is fitted once.

Without a ``hyperparameters`` block the classifier uses the package default
(``settings.DEFAULT_HYPERPARAMETERS``).
With ``tune_hyperparameters: True``, a grid search runs on the training rows first,
with inner folds grouped by polygon.

Projection configuration file
-----------------------------

Per projection, a separate configuration file is needed.

.. code-block:: yaml

    general:
      # must be absolute: project.py reads this file from the deploy folder
      input_dir: /absolute/path/to/example_data

    data:
      extent:
        file: path/to/polygons.gpkg
      indicators:
        precipitation:
          file: path/to/precipitation_projection.nc
          log: False
          stat: mean
        temperature:
          file: path/to/temperature_projection.nc
          log: False
          stat: min

The indicators must be the same, in the same order, as in the reference file.
The model is fitted on column positions, and the projection stops if names or order differ.

.. note::
    The projection data can be in the same files as the reference data or in separate files.
    Make sure reference and projection data are consistent and biases are removed.
