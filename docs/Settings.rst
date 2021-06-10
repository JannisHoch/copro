.. _settings:

Settings
=========================

The cfg-file
----------------

The main model settings need to be specified in a configuration file (``cfg-file``). 
This file looks like this, taken from the example run and data.

.. code-block:: console

    [general]
    input_dir=./example_data
    output_dir=./OUT
    # 1: all data; 2: leave-one-out model; 3: single variable model; 4: dubbelsteenmodel
    # Note that only 1 supports sensitivity_analysis
    model=1
    verbose=True

    [settings]
    # start year
    y_start=2000
    # end year
    y_end=2012

    [PROJ_files]
    # cfg-files
    proj_nr_1=./example_settings_proj.cfg

    [pre_calc]
    # if nothing is specified, the XY array will be stored in output_dir
    # if XY already pre-calculated, then provide path to npy-file
    XY=

    [extent]
    shp=waterProvinces/waterProvinces_Africa_eliminatedPolysLE20000km2.shp

    [conflict]
    # either specify path to file or state 'download' to download latest PRIO/UCDP dataset
    conflict_file=UCDP/ged201.csv
    min_nr_casualties=1
    # 1=state-based armed conflict; 2=non-state conflict; 3=one-sided violence
    type_of_violence=1,2,3

    [climate]
    shp=KoeppenGeiger/2000/Koeppen_Geiger_1976-2000.shp
    # define either one or more classes (use abbreviations!) or specify nothing for not filtering
    zones=BWh,BSh
    code2class=KoeppenGeiger/classification_codes.txt

    [data]
    # specify the path to the nc-file, whether the variable shall be log-transformed (True, False), and which statistical function should be applied
    # these three settings need to be separated by a comma
    # NOTE: variable name here needs to be identical with variable name in nc-file
    # NOTE: only statistical functions supported by rasterstats are valid
    precipitation=hydro/precipitation_monthTot_output_2000-01-31_to_2015-12-31_Africa_yearmean.nc,True,mean
    temperature=hydro/temperature_monthAvg_output_2000-01-31_to_2015-12-31_Africa_yearmean.nc,True,mean
    gdp=gdp/gdp_Africa.nc,True,mean

    [machine_learning]
    # choose from: MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
    scaler=QuantileTransformer
    # choose from: NuSVC, KNeighborsClassifier, RFClassifier
    model=RFClassifier
    train_fraction=0.7
    # number of repetitions
    n_runs=10

The specifics
----------------

Here, the different sections are explained briefly. 

**[general]**

- *input_dir*: (relative) path to the directory where the input data is stored. This requires all input data to be stored in one main folder;
- *output_dir*: (relative) path to the directory where output will be stored. If the folder does not exist yet, it will be created;
- *model*: the type of simulation to be run can be specified here. Currently, for different models are available:

    1. 'all data': all variable values are used to fit the model and predict results;
    2. 'leave one out': values of each variable are left out once, resulting in n-1 runs with n being the number of variables. This model can be used to identify the relative influence of one variable within the variable set;
    3. 'single variables': each variable is used as sole predictor once. With this model, the explanatory power of each variable on its own can be assessed;
    4. 'dubbelsteen': the relation between variables and conflict are abolished by shuffling the binary conflict data randomly. By doing so, the lower boundary of the model can be estimated.

.. note::

    All model types except ``all_data`` will be deprecated in a future release.

- *verbose*: if True, additional messages will be printed.

**[settings]**

- *y_start*: the start year of the simulation;
- *y_end*: the end year of the simulation. All data between y_start and y_end will be used to train and test the model;
- *n_runs*: the number repetitions of the split-sample test for training and testing the model. By repeating these steps multiple times, coincidental results can be avoided.

**[pre_calc]**

- *XY*: if the XY-data was already pre-computed in a previous run and stored as npy-file, it can be specified here and will be loaded from file. If nothing is specified, the model will save the XY-data by default to the output directory as ``XY.npy``;
- *clf*: path to the pickled fitted classifier from the reference run. Needed for projection runs only!

**[extent]**

- *shp*: the provided shape-file defines the area for which the model is applied. At the same time, it also defines at which aggregation level the output is determined.

.. note:: 

    The shp-file should contain multiple polygons covering the study area. Their size defines the output aggregation level. It is also possible to provide only one polygon, but model behaviour is not well tested for this case.

**[conflict]**

- *conflict_file*: path to the csv-file containing the conflict dataset. It is also possible to define 'download', then the latest conflict dataset is downloaded and used as input;
- *min_nr_casualties*: minimum number of reported casualties required for a conflict to be considered in the model;
- *type_of_violence*: the types of violence to be considered can be specified here. Multiple values can be specified. Types of violence are:

    1. state-based armed conflict: a contested incompatibility that concerns government and/or territory where the use of armed force between two parties, of which at least one is the government of a state, results in at least 25 battle-related deaths in one calendar year;
    2. non-state conflict: the use of armed force between two organized armed groups, neither of which is the government of a state, which results in at least 25 battle-related deaths in a year;
    3. one-sided violence: the deliberate use of armed force by the government of a state or by a formally organized group against civilians which results in at least 25 deaths in a year.

.. important::

    CoPro currently only works with UCDP data. As other data sources will be supported in the future, the conflict selection process will be come more elaborated.

**[climate]**

- *shp*: the provided shape-file defines the areas of the different Köppen-Geiger climate zones;
- *zones*: abbreviations of the climate zones to be considered in the model. Can either be 'None' or one or multiple abbreviations;
- *code2class*: converting the abbreviations to class-numbers used in the shp-file.

.. warning:: 

    The code2class-file should not be altered!

**[data]**

In this section, all variables to be used in the model need to be provided. The main convention is that the name of the file agrees with the variable name in the file. Only netCDF-files with annual data are supported.

For example, if the variable precipitation is provided in a file, this should be noted as follows

    [data]
    precipitation=/path/to/file/precipitation_file.nc

**[machine_learning]**

- *scaler*: the scaling algorithm used to scale the variable values to comparable scales. Currently supported are ``MinMaxScaler``, ``StandardScaler``, ``RobustScaler``, and ``QuantileTransformer``;
- *model*: the machine learning algorithm to be applied. Currently supported are ``NuSVC``, ``KNeighborsClassifier``, and ``RFClassifier``;
- *train_fraction*: the fraction of the XY-data to be used to train the model. The remaining data (1-train_fraction) will be used to predict and evaluate the model.