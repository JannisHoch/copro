Using Copro
============

Model workflow
---------------

`copro` trains a Random Forest classifier to estimate the risk of conflict occurrence.
To that end, it needs conflict data (from UCDP) and a set of features describing potential conflict drivers.
The temporal resolution of the model is annual: feature data should contain annual data too, and conflict is estimated for each year.

Besides the indicators, four features are built from the conflict data itself:
conflict in the polygon in the previous year, conflict in a neighbouring polygon in the previous year,
and the time since the polygon's last conflict, expressed as a decay with half-lives of 2 and 8 years.

Work happens in two stages.

**Evaluation.** The command line script trains the model on a training set and tests it on held-out rows.
The `split_strategy` key in the `machine_learning` block of the :ref:`Reference configuration file`
decides which rows are held out.
Several model instances can be run with different splits via `n_runs`.
The held-out rows of every run are written to a table, so performance can be analysed per row.

**Deployment.** The model used for projections is fitted once more, on all rows, by a separate script.
Projections, score-to-probability rules and feature attribution all use this deploy model.
The conflict state at the end of the reference period is the starting point of each projection.

Via the command line
---------------------
The most convenient way to use `copro` is via the command line.
All model settings are given in one configuration file.

.. toctree::
   :maxdepth: 1

   cli.rst
   config.rst
   scripts.rst

API
----
For bespoke applications, it is possible to use `copro` as a library.
Please find more information in the :ref:`API documentation`.
