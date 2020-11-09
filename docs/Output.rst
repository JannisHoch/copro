Output
=========================

The model can produce a range of output files. Output is stored in the output folder as specified in the configurations-file (cfg-file).

.. note:: 

    In addition to these output files, the model settings file (cfg-file) is automatically copied to the output folder.

.. important:: 

    Not all model types provide the output mentioned below. If the 'leave-one-out' or 'single variable' model are selected, only the metrics are stored to a csv-file.

.. important::

    Most of the output can only be produced when running a reference model, i.e. when comparing the predictions against observations. 
    If running a prediction model, only the chance of conflict per polygon is stored to file.

Selected polygons
------------------
A shp-file named ``selected_polygons.shp`` contains all polygons after performing the selection procedure.

Selected conflicts
-------------------
The shp-file ``selected_conflicts.shp`` contains all conflict data points after performing the selection procedure.

Sampled variable and conflict data
-----------------------------------
During model execution, data is sampled per polygon and time step. 
This data contains the geometry and ID of each polygon as well as unscaled variable values (X) and a boolean identifier whether conflict took place or not (Y).
If the model is re-run without making changes to the data and how it is sampled, the resulting XY-array is stored to ``XY.npy``. This file can be loaded again with ``np.load()``.

.. note:: 

    Note that ``np.load()`` returns an array. This can be further processed with e.g. pandas.

ML classifier
--------------
At the end of a reference run, the chosen classifier is fitted with all available XY-data.
To be able to re-use the classifier (e.g. to make predictions), it is pickled to ``clf.pkl``.

All predictions
------------------
Per model run, a fraction of the total XY-data is used to make a prediction. 
To be able to analyse model output, all predictions (stored as pandas dataframes) made per run are appended to a main output-dataframe.
This dataframe is, actually, the basis of all futher analyes.
When storing to file, this can become a rather large file. 
Therefore, the dataframe is converted to npy-file (``out_y_df.npy``). This file can be loaded again with ``np.load()``.

.. note:: 

    Note that ``np.load()`` returns an array. This can be further processed with e.g. pandas.

Prediction metrics
-----------------------
Per model run, a range of metrics are computed to evalute the predictions made. They are all appended to a dictionary and saved to the file ``out_dict.csv``.

Model prediction per polygon
---------------------------
At the end of all model repetitions, the resulting output dataframe contains multiple predictions for each polygon.
By aggregating results per polygon, it is possible to assess model output spatially. 

Three main output metrics are calculated per polygon:

1. The chance of a correct (*CCP*), defined as the ratio of number of correct predictions made to overall number of predictions made;
2. The total number of conflicts in the test  (*NOC*);
3. The chance of conflict (*COC*), defined as the ration of number of conflict predictions to overall number of predictions made.

k-fold analysis
^^^^^^^^^^^^^^^^
The model is repeated several times to eliminate the influence of how the data is split into training and test samples.
As such, the accuracy per run and polygon will differ.

To account for that, the resulting data set containing all predictions at the end of the run is split in k chunks. 
Subsequently, the mean, median, and standard deviation of CCP is determined from the k chunks.

The resulting shp-file is named ``kFold_CCP_stats.shp``.

all data
^^^^^^^^^

All output metrics (CCP, NOC, COC) are determined based on the entire data set at the end of the run, i.e. without splitting it in chunks.

The data is stored to ``all_stats.shp``.

.. note::

    In addition to these shp-file, various plots can be stored by using the provided plots-functions. The plots aer stored in the output directory too.
    Note that the plot settings cannot yet be fully controlled via those functions, i.e. it is more anticipated for debugging.
    To create custom-made plots, rather use the shp-files and csv-file.



