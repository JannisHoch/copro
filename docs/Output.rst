Output
=========================

The model can produce a range of output files. All output is stored in the output folder as specified in the configurations-file (cfg-file).

In addition to the output files listed below, the model settings file (cfg-file) is automatically copied to the output folder.

.. important:: 

    Not all model types provide the output mentioned below. If the 'leave-one-out' or 'single variable' model are selected, only the metrics are stored to a csv-file.

List of output files
---------------------------

+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| File name                     | Description                                                                                 | Note                                                                                        |
+===============================+=============================================================================================+=============================================================================================+
| ``selected_polygons.shp``     | Shapefile containing all remaining polygons after selection procedure                       |                                                                                             |
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``selected_conflicts.shp``    | Shapefile containing all remaining conflict points after selection procedure                |                                                                                             | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``XY.npy``                    | NumPy-array containing geometry, ID, and scaled data of sample (X) and target data (Y)      | can be provided in cfg-file to safe time in next run; file can be loaded with numpy.load()  | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``X.npy``                     | NumPy-array containing geometry, ID, and scaled data of sample (X)                          | only written in projection run; file can be loaded with numpy.load()                        | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``clf.pkl``                   | Pickled classifier fitted with the entirety of XY-data                                      | needed to perform projection run; file can be loaded with pickle.load()                     | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``raw_output_data.npy``       | NumPy-array containing each single prediction made in the reference run                     | will contain multiple predictions per polygon; file can be loaded with numpy.load()         | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``evaluation_metrics.csv``    | Various evaluation metrics determined per repetition of the split-sample test repetition    | file can e.g. be loaded with pandas.read_csv()                                              | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``ROC_data_tprs.csv``         | False-positive rates per repetition of the split-sample test repetition                     | file can e.g. be loaded with pandas.read_csv(); data can be used to later plot ROC-curve    | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``ROC_data_aucs.csv``         | Area-under-curve values per repetition of the split-sample test repetition                  | file can e.g. be loaded with pandas.read_csv(); data can be used to later plot ROC-curve    | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
| ``output_per_polygon.shp``    | Shapefile containing resulting conflict risk estimates per polygon                          | for further explanation, see below                                                          | 
+-------------------------------+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+

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




