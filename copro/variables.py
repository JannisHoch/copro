import xarray as xr
import rasterio as rio
import pandas as pd
import geopandas as gpd
import rasterstats as rstats
import numpy as np
import os, sys
import math
from distutils import util

import warnings
warnings.filterwarnings("ignore")

def nc_with_float_timestamp(extent_gdf, config, root_dir, var_name, sim_year):
    """This function extracts a statistical value from a netCDF-file (specified in the config-file) for each polygon specified in extent_gdf for a given year.
    By default, the mean value of all cells within a polygon is computed.
    The resulting list does not contain additional meta-information about the files or polygons and is mostly intended for data-driven approaches such as machine learning.

    NOTE:
    The var_name must be identical to the key in the config-file. 

    NOTE:
    This function is specifically written for netCDF-files where the time variable contains integer (year-)values, e.g. 1995, 1996, ...

    NOTE:
    Works only with nc-files with annual data.

    Args:
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
        config (config): parsed configuration settings of run.
        root_dir (str): path to location of cfg-file. 
        var_name (str): name of variable in nc-file, must also be the same under which path to nc-file is specified in cfg-file.
        sim_year (int): year for which data is extracted.

    Raises:
        ValueError: raised if the extracted variable at a time step does not contain data

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as extent_gdf
    """   

    data_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name)).rsplit(',')

    if len(data_fo) != 3:
        raise ValueError('ERROR: not all settings for input data set {} provided - it must contain of path, False/True, and statistical method'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name))))
    else:
        nc_fo = data_fo[0]
        ln_flag = bool(util.strtobool(data_fo[1]))
        stat_method = str(data_fo[2])

    if config.getboolean('general', 'verbose'): 
        if ln_flag:
            print('DEBUG: calculating log-transformed {0} {1} per aggregation unit from file {2} for year {3}'.format(stat_method, var_name, nc_fo, sim_year))
        else:
            print('DEBUG: calculating {0} {1} per aggregation unit from file {2} for year {3}'.format(stat_method, var_name, nc_fo, sim_year))

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]
    if ln_flag:
        nc_var = np.log(nc_var)
        if config.getboolean('general', 'verbose'): print('DEBUG: log-transform variable {}'.format(var_name))
    # open nc-file with rasterio to get affine information
    affine = rio.open(nc_fo).transform

    # get values from data-array for specified year
    nc_arr = nc_var.sel(time=sim_year)
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('ERROR: the data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    # initialize output list
    list_out = []
    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(extent_gdf)):

        # province i
        prov = extent_gdf.iloc[i]

        # compute zonal stats for this province
        zonal_stats = rstats.zonal_stats(prov.geometry, nc_arr_vals, affine=affine, stats=stat_method)
        val = zonal_stats[0][stat_method]

        # # if specified, log-transform value
        if ln_flag:
            # works only if zonal stats is not None, i.e. if it's None it stays None
            if val != None: val = np.log(val)
        
        # in case log-transformed value results in -inf, replace with None
        if val == -math.inf:
            if config.getboolean('general', 'verbose'): print('INFO: set -inf to None')
            val = None

        # print a warning if result is None
        if (val == None) and (config.getboolean('general', 'verbose')): 
            print('WARNING: NaN computed!')

        list_out.append(val)

    if config.getboolean('general', 'verbose'): print('DEBUG: ... done.')

    return list_out

def nc_with_continous_datetime_timestamp(extent_gdf, config, root_dir, var_name, sim_year):
    """This function extracts a statistical value from a netCDF-file (specified in the config-file) for each polygon specified in extent_gdf for a given year.
    By default, the mean value of all cells within a polygon is computed.
    The resulting list does not contain additional meta-information about the files or polygons and is mostly intended for data-driven approaches such as machine learning.

    NOTE:
    The var_name must be identical to the key in the config-file. 

    NOTE:
    Works only with nc-files with annual data.

    Args:
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted
        config (config): parsed configuration settings of run.
        root_dir (str): path to location of cfg-file. 
        var_name (str): name of variable in nc-file, must also be the same under which path to nc-file is specified in cfg-file.
        sim_year (int): year for which data is extracted.

    Raises:
        ValueError: raised if specfied year cannot be found in years in nc-file
        ValueError: raised if the extracted variable at a time step does not contain data

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as extent_gdf
    """   

    data_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name)).rsplit(',')

    if len(data_fo) != 3:
        raise ValueError('ERROR: not all settings for input data set {} provided - it must contain of path, False/True, and statistical method'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name))))
    else:
        nc_fo = data_fo[0]
        ln_flag = bool(util.strtobool(data_fo[1]))
        stat_method = str(data_fo[2])

    if config.getboolean('general', 'verbose'): 
        if ln_flag:
            print('DEBUG: calculating log-transformed {0} {1} per aggregation unit from file {2} for year {3}'.format(stat_method, var_name, nc_fo, sim_year))
        else:
            print('DEBUG: calculating {0} {1} per aggregation unit from file {2} for year {3}'.format(stat_method, var_name, nc_fo, sim_year))

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]
    # get years contained in nc-file as integer array to be compatible with sim_year
    years = pd.to_datetime(nc_ds.time.values).to_period(freq='Y').strftime('%Y').to_numpy(dtype=int)
    if sim_year not in years:
        raise ValueError('ERROR: the simulation year {0} can not be found in file {1}'.format(sim_year, nc_fo))
    
    # get index which corresponds with sim_year in years in nc-file
    sim_year_idx = int(np.where(years == sim_year)[0])
    # get values from data-array for specified year based on index
    nc_arr = nc_var.sel(time=nc_ds.time.values[sim_year_idx])
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('ERROR: no data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    # open nc-file with rasterio to get affine information
    affine = rio.open(nc_fo).transform

    # initialize output list
    list_out = []
    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(extent_gdf)):

        # province i
        prov = extent_gdf.iloc[i]

        # compute zonal stats for this province
        zonal_stats = rstats.zonal_stats(prov.geometry, nc_arr_vals, affine=affine, stats=stat_method)
        val = zonal_stats[0][stat_method]

        # # if specified, log-transform value
        if ln_flag:
            # works only if zonal stats is not None, i.e. if it's None it stays None
            if val != None: val = np.log(val)
        
        # in case log-transformed value results in -inf, replace with None
        if val == -math.inf:
            if config.getboolean('general', 'verbose'): print('INFO: set -inf to None')
            val = None

        # print a warning if result is None
        if (val == None) and (config.getboolean('general', 'verbose')): 
            print('WARNING: NaN computed!')

        list_out.append(val)

    if config.getboolean('general', 'verbose'): print('DEBUG: ... done.')

    return list_out