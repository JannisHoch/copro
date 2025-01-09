Model configuration file
========================

.. important:: 
    The here described configuration file is compatible with CoPro version < 2.0.0.
    For versions >= 2.0.0, the configuration file needs to be realigned with the new requirements.

The command line script takes on argument, which is a model configuration file.
This file covers all necessary information to run the model for the reference period.

In case projections should be made, each projection is specified in a separate (reduced) configuration file.
Multiple files can be specified.
The name of each projection is specified via the key in the section `[PROJ_files]`.

.. note::
    The file extension of the configuration files is not important -  we use `.cfg`.

Reference configuration file
----------------------------

The configuration file for the reference period needs to contain the following sections.

.. note::
    All paths should be relative to `input_dir`.

.. code-block::

    [general]
    input_dir=./example_data
    output_dir=./OUT

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
    shp=path/to/polygons.shp

    [conflict]
    # PRIO/UCDP dataset
    conflict_file=path/to/ged201.csv
    min_nr_casualties=1
    # 1=state-based armed conflict; 2=non-state conflict; 3=one-sided violence
    type_of_violence=1,2,3

    [data]
    # specify the path to the nc-file, whether the variable shall be log-transformed (True, False), and which statistical function should be applied
    # these three settings need to be separated by a comma
    # NOTE: variable name here needs to be identical with variable name in nc-file
    # NOTE: only statistical functions supported by rasterstats are valid
    precipitation=path/to/precipitation_data.nc,True,mean
    temperature=path/to/temperature_data.nc,True,min
    gdp=path/to/gdp_data.nc,False,max

    [machine_learning]
    # choose from: MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
    scaler=QuantileTransformer
    train_fraction=0.7
    # number of model instances
    n_runs=10

Projection configuration file
------------------------------

Per projection, a separate configuration file is needed.
This file needs to contain the following sections.

.. code-block:: 
    
    [general]
    input_dir=./example_data
    verbose=True

    [settings]
    # end year of projections
    y_proj=2015

    [pre_calc]
    # if nothing is specified, the XY array will be stored in output_dir
    # if XY already pre-calculated, then provide (absolute) path to npy-file
    XY=

    [data]
    # specify the path to the nc-file, whether the variable shall be log-transformed (True, False), and which statistical function should be applied
    # these three settings need to be separated by a comma
    # NOTE: variable name here needs to be identical with variable name in nc-file
    # NOTE: only statistical functions supported by rasterstats are valid
    precipitation=path/to/precipitation_data.nc,True,mean
    temperature=path/to/temperature_data.nc,True,min
    gdp=path/to/gdp_data.nc,False,max

.. note::
    The projection data can be in the same file as the reference data or in separate files.
    Note that it's important to ensure reference and projection data are consistent and biases are removed.

