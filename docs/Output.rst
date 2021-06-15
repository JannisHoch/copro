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

+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| File name                     | Description                                                                                 | Note                                                                                        |
+===============================+=============================================================================================+=============================================================================================+
| ``selected_polygons.shp``     | Shapefile containing all remaining polygons after selection procedure                       |                                                                                             |
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``selected_conflicts.shp``    | Shapefile containing all remaining conflict points after selection procedure                |                                                                                             | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``XY.npy``                    | NumPy-array containing geometry, ID, and scaled data of sample (X) and target data (Y)      | can be provided in cfg-file to safe time in next run; file can be loaded with numpy.load()  | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``raw_output_data.npy``       | NumPy-array containing each single prediction made in the reference run                     | will contain multiple predictions per polygon; file can be loaded with numpy.load()         | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``evaluation_metrics.csv``    | Various evaluation metrics determined per repetition of the split-sample tests              | file can e.g. be loaded with pandas.read_csv()                                              | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``feature_importance.csv``    | Importance of each model variable in making projections                                     | this is a property of RF Classifiers and thus only obtainable if RF Classifier is used      |                                                                               | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``permutation_importance.csv``| Mean permutation importance per model variable                                              | computed with sklearn.inspection.permutation_importance_                                    | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``ROC_data_tprs.csv``         | False-positive rates per repetition of the split-sample test repetition                     | file can e.g. be loaded with pandas.read_csv(); data can be used to later plot ROC-curve    | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``ROC_data_aucs.csv``         | Area-under-curve values per repetition of the split-sample test repetition                  | file can e.g. be loaded with pandas.read_csv(); data can be used to later plot ROC-curve    | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``output_for_REF.geojson``    | GeoJSON-file containing resulting conflict risk estimates per polygon                       | based on out-of-sample projections of _REF run                                              | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+

.. _sklearn.inspection.permutation_importance: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html

_PROJ
^^^^^^

Per projection, CoPro creates one output file per projection year.

+------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
| File name              | Description                                                                                 | Note                                                                            |
+========================+=============================================================================================+=================================================================================+
| ``output_in_YEAR``     | GeoJSON-file containing model output per polygon averaged over all classifier instances     | number of instances is set with ``n_runs`` in ``[machine_learning]`` section    |                                                                             |
+------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+

Conflict risk per polygon
---------------------------

At the end of all model repetitions, the resulting output data frame contains multiple predictions for each polygon.
By aggregating results per polygon, it is possible to assess model output spatially. 

Three main output metrics are calculated per polygon and saved to ``output_per_polygon.shp``:

1. The number of predictions made per polygon;
2. The number of observed conflicts per polygon;
3. The number of predicted conflicts per polygon;
4. The fraction of correct predictions (*FOP*), defined as the ratio of the number of correct predictions over the total number of predictions made;
5. The chance of conflict (*COC*), defined as the ration of the number of conflict predictions over the total number of predictions made.

.. important::

    For projection runs, only the COC can be determined as no conflict observations are used/available.




