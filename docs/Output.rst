Output
=========================

The model produces a range of output files by default. Output is stored in the output folder.

.. note:: 

    In addition to these output files, the model settings file (cfg-file) is automatically copied to the output folder.

Selected polygons
------------------
A shp-file named ``selected_polygons.shp`` contains all polygons after performing the selection procedure.

Selected conflicts
-------------------
The shp-file ``selected_conflicts.shp`` contains all conflict data points after performing the selection procedure.

Overall output
---------------

Per model run, a fraction of the total XY-data is used to make a prediction. 
To be able to analyse model output, all predictions made per run are appended to a main output-file.
This file is, actually, the basis of all futher analyes.
Since this can become a rather large file, the dataframe is converted to npy-file (``out_y_df.npy``). This file can be loaded again with ``np.load()``.

.. note:: 

    Note that ``np.load()`` returns an array. This can be further processed with e.g. pandas.

Model accuracy per run
-----------------------

Per model run, a range of metrics are computed. They are all appended to a dictionary and saved to the file ``out_dict.csv``.

Model accuracy per polygon
---------------------------

At the end of all model repetitions, the resulting output contains multiple predictions for each polygon.
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



