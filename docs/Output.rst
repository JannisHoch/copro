Output
=========================

Output folder structure
---------------------------

All output is stored in the output folder as specified in the configurations-file (cfg-file) under [general].

.. code-block:: console

    [general]
    output_dir=./path/to/store/output

By default, CoPro creates two sub-folders: ``_REF`` and ``_PROJ``. In the latter, another sub-folder will be created per projection defined in the cfg-file.
In the example below, this would be the folders ``/_PROJ/SSP1`` and ``/_PROJ/SSP2``.

.. code-block:: console

    [PROJ_files]    
    SSP1=/path/to/ssp1.cfg
    SSP2=/path/to/ssp2.cfg

List of output files
---------------------------

.. important:: 

    Not all model types provide the output mentioned below. If the 'leave-one-out' or 'single variable' model are selected, only the metrics are stored to a csv-file.

_REF
^^^^^^

In addition to the output files listed below, the cfg-file is automatically copied to the _REF folder.

``selected_polygons.shp``: Shapefile containing all remaining polygons after selection procedure.

``selected_conflicts.shp``: Shapefile containing all remaining conflict points after selection procedure,

``XY.npy``: NumPy-array containing geometry, ID, and scaled data of sample (X) and target data (Y). 
Can be provided in cfg-file to safe time in next run; file can be loaded with numpy.load().

``raw_output_data.npy``: NumPy-array containing each single prediction made in the reference run.
Will contain multiple predictions per polygon. File can be loaded with numpy.load().

``evaluation_metrics.csv``: Various evaluation metrics determined per repetition of the split-sample tests.
File can e.g. be loaded with pandas.read_csv().

``feature_importance.csv``: Importance of each model variable in making projections.
This is a property of RF Classifiers and thus only obtainable if RF Classifier is used.

``permutation_importance.csv``: Mean permutation importance per model variable.
Computed with sklearn.inspection.permutation_importance_.

``ROC_data_tprs.csv`` and ``ROC_data_aucs.csv``: False-positive rates respectively Area-under-curve values per repetition of the split-sample test.
Files can e.g. be loaded with pandas.read_csv() and can be used to later plot ROC-curve.

``output_for_REF.geojson``: GeoJSON-file containing resulting conflict risk estimates per polygon based on out-of-sample projections of _REF run.

.. _sklearn.inspection.permutation_importance: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html

Conflict risk per polygon
""""""""""""""""""""""""""

At the end of all model repetitions, the resulting ``raw_output_data.npy`` file contains multiple out-of-sample predictions per polygon.
By aggregating results per polygon, it is possible to assess model output spatially as stored in ``output_for_REF.geojson``. 

The main output metrics are calculated per polygon and saved to ``output_per_polygon.shp``:

1. nr_predictions: the number of predictions made;
2. nr_correct_predictions: the number of correct predictions made;
3. nr_observed_conflicts: the number of observed conflict events;
4. nr_predicted_conflicts: the number of predicted conflicts;
5. min_prob_1: minimum probability of conflict in all repetitions;
6. probability_of_conflict (POC): probability of conflict averaged over all repetitions;
7. max_prob_1: maximum probability of conflict in all repetitions;
8. fraction_correct_predictions (FOP): ratio of the number of correct predictions over the total number of predictions made;
9. chance_of_conflict: ratio of the number of conflict predictions over the total number of predictions made.

_PROJ
^^^^^^

Per projection, CoPro creates one output file per projection year.

``output_in_<YEAR>``: GeoJSON-file containing model output per polygon averaged over all classifier instances per YEAR of the projection.
The number of instances is set with ``n_runs`` in ``[machine_learning]`` section.

Conflict risk per polygon
""""""""""""""""""""""""""

During the projection run, each classifier instances produces its own output per YEAR.
CoPro merges these outputs into one ``output_in_<YEAR>.geojson`` file. 

As there are no observations available for the projection period, the output metrics differ from the reference run:

1. nr_predictions: the number of predictions made, ie. number of classifier instances;
2. nr_predicted_conflicts: the number of predicted conflicts.
3. min_prob_1: minimum probability of conflict in all outputs of classifier instances.
4. probability_of_conflict (POC): probability of conflict averaged over all outputs of classifier instances.
5. max_prob_1: maximum probability of conflict in all outputs of classifier instances;
6. chance_of_conflict: ratio of the number of conflict predictions over the total number of predictions made.
