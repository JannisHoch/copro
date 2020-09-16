Model settings
=========================

The main model settings need to be specified in a configuration file. Here, the different sections are explained briefly. 
Additionally, the cfg-file included in the GitHub-repository already contains the key information.

.. note::

    The file-type (we here refer to the cfg-file) is less relevant as long as the file can be parsed with ConfigParser.

**[general]**

- *input_dir*: (relative) path to the directory where the input data is stored. This requires all input data to be stored in one main folder;
- *output_dir*: (relative) path to the directory where output will be stored. If folder does not exist yet, it will be created;
- *model*: the type of simulation to be run can be specified here. Currently, for different models are available:

    1. 'all data': all variable values are used to fit the model and predict results;
    2. 'leave one out': values of each variable are left out once, resulting in n-1 runs with n being the number of variables. This model can be used to identify the relative influence of one variable within the variable set;
    3. 'single variables': each variable is used as sole predictor once. With this model, the explanatory power of each variable on its own can be assessed;
    4. 'dubbelsteen': the relation between variables and conflict are abolished by shuffling the binary conflict data randomly. By doing so, the lower boundary of the model can be estimated.

.. note::

    the 'leave_one_out' and 'single_variables' models are only tested in beta-state.

- *verbose*: if True, model messages will be printed to screen.

**[settings]**

- *y_start*: the start year of the simulation;
- *y_end*: the end year of the simulation. All data between y_start and y_end will be used to train and test the model;
- *n_runs*: the number of model executions. By repeating the split, train, test steps multiple times, coincidental results can be avoided.

**[pre_calc]**

- *XY*: if the XY-data was already pre-computed in a previous run and stored as npy-file, it be specified here and will be loaded from file. If nothing is specified, the model will save the XY-data by default to the input directory.

**[extent]**

- *shp*: the provided shape-file defines the area for which the model is applied. At the same time, it also defines at which aggregation level the output is determined.

.. note:: 

    The shp-file should contain multiple polygons covering the study area. Their size defines the output aggregation level. It is also possible to provide only one polygon, but model behaviour is not well tested for this case.

**[conflict]**

- *conflict_file*: path to the csv-file containing the conflict dataset. It is also possible to define 'download', then the latest conflict dataset is downloaded and used as input;
- *min_nr_casualties*: minimum number of reported casualties required for a conflict to be considered in the model;
- *type_of_violence*: the types of violence to be considered can be specified here. Multiple values can be specified. Types of violence are:

    1. state-based armed conflict: a contested incompatibility that concerns government and/or territory where the use of armed force between two parties, of which at least one is the government of a state, results in at least 25 battle-related deaths in one calendar year;
    2. non-state conflict: the use of armed force between two organised armed groups, neither of which is the government of a state, which results in at least 25 battle-related deaths in a year;
    3. one-sided violence: the deliberate use of armed force by the government of a state or by a formally organised group against civilians which results in at least 25 deaths in a year.

.. note::

    Currently, the model works only with data from the `PRIO/UCDP programme <https://www.prio.org/Data/Armed-Conflict/UCDP-PRIO/>`_. Other conflict datasets such `ACLED <https://acleddata.com/>`_ are not supported due to differences in data categorization.

**[climate]**

- *shp*: the provided shape-file defines the areas of the different Köppen-Geiger climate zones;
- *zones*: abbreviations of the climate zones to be considered in the model. Can either be 'None' or one or multiple abbreviations;
- *code2class*: converting the abbreviations to class-numbers used in the shp-file.

.. warning:: 

    The code2class-file should not be altered except if another shp-file is provided containing undocumented climate classes.

**[env_vars]**

In this section, all variables to be used in the model need to be provided. The main convention is that the name of the file agrees with the variable name in the file. Only netCDF-files with annual data are supported.

**[machine_learning]**

- *scaler*: the scaling algorithm used to scale the variable values to comparable scales. Currently supported are ``MinMaxScaler``, ``StandardScaler``, ``RobustScaler``, and ``QuantileTransformer``;
- *model*: the machine learning algorithm to be applied. Currently supported are ``NuSVC``, ``KNeighborsClassifier``, and ``RFClassifier``;

.. note:: 

    The parameters of the models were already calibrated using the sklearn `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.

- *train_fraction*: the fraction of the XY-data to be used to train the model. The remaining data (1-train_fraction) will be used to predict and evaluate the model.