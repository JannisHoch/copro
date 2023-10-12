.. _settings:

Settings
=========================

The cfg-file
----------------

The main model settings need to be specified in a configuration file (``cfg-file``). 
This file looks like this.

.. code-block:: console

    [general]
    input_dir=./path/to/input_data
    output_dir=./path/to/store/output
    verbose=True

    [settings]
    # start year
    y_start=2000
    # end year
    y_end=2012

    [PROJ_files]
    # cfg-files
    proj_nr_1=./path/to/projection/settings_proj.cfg

    [pre_calc]
    # if nothing is specified, the XY array will be stored in output_dir
    # if XY already pre-calculated, then provide path to npy-file
    XY=

    [extent]
    shp=folder/with/polygons.shp

    [conflict]
    # either specify path to file or state 'download' to download latest PRIO/UCDP dataset
    conflict_file=folder/with/conflict_data.csv
    min_nr_casualties=1
    # 1=state-based armed conflict. 2=non-state conflict. 3=one-sided violence
    type_of_violence=1,2,3

    [climate]
    shp=folder/with/climate_zones.shp
    # define either one or more classes (use abbreviations!) or specify nothing for not filtering
    zones=
    code2class=folder/with/classification_codes.txt

    [data]
    # specify the path to the nc-file, whether the variable shall be log-transformed (True, False), and which statistical function should be applied
    # these three settings need to be separated by a comma
    # NOTE: variable name here needs to be identical with variable name in nc-file
    # NOTE: only statistical functions supported by rasterstats are valid
    precipitation=folder/with/precipitation_data.nc,False,mean
    temperature=folder/with/temperature_data.nc,False,mean
    population=folder/with/population_data.nc,True,sum

    [machine_learning]
    # choose from: MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
    scaler=QuantileTransformer
    # choose from: RFClassifier or RFRegression
    model=RFClassifier
    train_fraction=0.7
    # number of repetitions
    n_runs=10

.. note::

    All paths for ``input_dir``, ``output_dir``, and in ``[PROJ_files]`` are relative to the location of the cfg-file. 

.. important::

    Empty spaces should be avoided in the cfg-file, besides for those lines commented out with '#'.

The sections
----------------

Here, the different sections are explained briefly. 

[general]
^^^^^^^^^^^^^^^^

``input_dir``: (relative) path to the directory where the input data is stored. This requires all input data to be stored in one main folder, sub-folders are possible.

``output_dir``: (relative) path to the directory where output will be stored. 
If the folder does not exist yet, it will be created. 
CoPro will automatically create the sub-folders ``_REF`` for output for the reference run, and ``_PROJ`` for output from the (various) projection runs.

``verbose``: if True, additional messages will be printed.

[settings]
^^^^^^^^^^^^^^^^

``y_start``: the start year of the reference run.

``y_end``: the end year of the reference run. 
The period between ``y_start`` and ``y_end`` will be used to train and test the model.

``y_proj``: the end year of the projection run.
The period between ``y_end`` and ``y_proj`` will be used to make annual projections.

[PROJ_files]
^^^^^^^^^^^^^^^^

A key section. Here, one (slightly different) cfg-file per projection needs to be provided. 
This way, multiple projection runs can be defined from within the "main" cfg-file.

The conversion is that the projection name is defined as value here.
For example, the projections "SSP1" and "SSP2" would be defined as

.. code-block:: console

    SSP1=/path/to/ssp1.cfg
    SSP2=/path/to/ssp2.cfg

A cfg-file for a projection is shorter than the main cfg-file used as command line argument and looks like this:

.. code-block:: console

    [general]
    input_dir=./path/to/input_data
    verbose=True

    [settings]
    # year for which projection is to be made
    y_proj=2050

    [data]
    # specify the path to the nc-file, whether the variable shall be log-transformed (True, False), and which statistical function should be applied
    # these three settings need to be separated by a comma
    # NOTE: variable name here needs to be identical with variable name in nc-file
    # NOTE: only statistical functions supported by rasterstats are valid
    precipitation=folder/with/precipitation_data.nc,False,mean
    temperature=folder/with/temperature_data.nc,False,mean
    population=folder/with/population_data.nc,True,sum

[pre_calc]
^^^^^^^^^^^^^^^^

``XY``: if the XY-data was already pre-computed in a previous run and stored as npy-file, it can be specified here and will be loaded from file to save time. 
If nothing is specified, the model will save the XY-data by default to the output directory as ``XY.npy``.

[extent]
^^^^^^^^^^^^^^^^

``shp``: the provided shape-file defines the boundaries for which the model is applied. 
At the same time, it also defines at which aggregation level the output is determined.

.. note:: 

    The shp-file can contain multiple polygons covering the study area. Their size defines the output aggregation level. It is also possible to provide only one polygon, but model behaviour is not well tested for this case.

[conflict]
^^^^^^^^^^^^^^^^

``conflict_file``: path to the csv-file containing the conflict dataset. 
It is also possible to define ``download``, then the latest conflict dataset (currently version 20.1) is downloaded and used as input.

``min_nr_casualties``: minimum number of reported casualties required for a conflict to be considered in the model.

``type_of_violence``: the types of violence to be considered can be specified here. 
Multiple values can be specified. Types of violence are:

    1. 'state-based armed conflict': a contested incompatibility that concerns government and/or territory where the use of armed force between two parties, of which at least one is the government of a state, results in at least 25 battle-related deaths in one calendar year.
    2. 'non-state conflict': the use of armed force between two organized armed groups, neither of which is the government of a state, which results in at least 25 battle-related deaths in a year.
    3. 'one-sided violence': the deliberate use of armed force by the government of a state or by a formally organized group against civilians which results in at least 25 deaths in a year.

.. important::

    CoPro currently only works with UCDP data.

[climate]
^^^^^^^^^^^^^^^^

``shp``: the provided shape-file defines the areas of the different Köppen-Geiger climate zones.

``zones``: abbreviations of the climate zones to be considered in the model.
Can either be 'None' or one or multiple abbreviations.

``code2class``: converting the abbreviations to class-numbers used in the shp-file.

.. warning:: 

    The code2class-file should not be altered!

[data]
^^^^^^^^^^^^^^^^

In this section, all variables to be used in the model need to be provided. 
The paths are relative to ``input_dir``.
Only netCDF-files with annual data are supported.

The main convention is that the name of the file agrees with the variable name in the file.
For example, if the variable ``precipitation`` is provided in a nc-file, this should be noted as follows

.. code-block:: console

    [data]
    precipitation=folder/with/precipitation_data.nc

CoPro furthermore requires information whether the values sampled from a file are ought to be log-transformed.

Besides, it is possible to define a statistical function that is applied when sampling from file per polygon of the ``shp-file``.
CoPro makes use of the ``zonal_stats`` function available within `rasterstats <https://pythonhosted.org/rasterstats/rasterstats.html>`_.

To determine the log-scaled mean value of precipitation per polygon, the following notation is required:

.. code-block:: console

    [data]
    precipitation=folder/with/precipitation_data.nc,False,mean

[machine_learning]
^^^^^^^^^^^^^^^^^^^^

``scaler``: the scaling algorithm used to scale the variable values to comparable scales. 
Currently supported are 

    - `MinMaxScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_;
    - `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_;
    - `RobustScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html>`_;
    - `QuantileTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html>`_.

``model``: the machine learning algorithm to be applied. 
Currently supported are 

    - `NuSVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html>`_; 
    - `KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_;
    - `RFClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

``train_fraction``: the fraction of the XY-data to be used to train the model. 
The remaining data (1-train_fraction) will be used to predict and evaluate the model.

``n_runs``: the number of classifiers to use.