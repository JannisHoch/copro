import xarray as xr
import rasterio as rio
import pandas as pd
import geopandas as gpd
import rasterstats as rstats
import numpy as np
import os, sys

import warnings
warnings.filterwarnings("ignore")

def nc_with_float_timestamp(extent_gdf, config, root_dir, var_name, sim_year, stat_func='mean'):
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
        stat_func (str, optional): Statistical function to be applied, choose from available options in rasterstats package. Defaults to 'mean'.

    Raises:
        ValueError: raised if the extracted variable at a time step does not contain data

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as extent_gdf
    """   
    # get path to netCDF-file.
    # nc_fo = os.path.join(os.path.abspath(config.get('general', 'input_dir')), 
    #                      config.get('data', var_name))

    nc_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name))

    if config.getboolean('general', 'verbose'): print('DEBUG: calculating mean {0} per aggregation unit from file {1} for year {2}'.format(var_name, nc_fo, sim_year))

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]

    # open nc-file with rasterio to get affine information
    affine = rio.open(nc_fo).transform

    # get values from data-array for specified year
    nc_arr = nc_var.sel(time=sim_year)
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('the data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    # initialize output list
    list_out = []
    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(extent_gdf)):
        prov = extent_gdf.iloc[i]
        zonal_stats = rstats.zonal_stats(prov.geometry, nc_arr_vals, affine=affine, stats=stat_func)
        if (zonal_stats[0][stat_func] == None) and (config.getboolean('general', 'verbose')): 
            print('WARNING: NaN computed!')
        list_out.append(zonal_stats[0][stat_func])

    if config.getboolean('general', 'verbose'): print('DEBUG: ... done.')

    return list_out

def nc_with_continous_datetime_timestamp(extent_gdf, config, root_dir, var_name, sim_year, stat_func='mean'):
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
        stat_func (str, optional): Statistical function to be applied, choose from available options in rasterstats package. Defaults to 'mean'.

    Raises:
        ValueError: raised if specfied year cannot be found in years in nc-file
        ValueError: raised if the extracted variable at a time step does not contain data

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as extent_gdf
    """   
    # get path to netCDF-file.
    # nc_fo = os.path.join(os.path.abspath(config.get('general', 'input_dir')), 
    #                      config.get('data', var_name))

    nc_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name))
    
    if config.getboolean('general', 'verbose'): print('DEBUG: calculating mean {0} per aggregation unit from file {1} for year {2}'.format(var_name, nc_fo, sim_year))

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]
    # get years contained in nc-file as integer array to be compatible with sim_year
    years = pd.to_datetime(nc_ds.time.values).to_period(freq='Y').strftime('%Y').to_numpy(dtype=int)
    if sim_year not in years:
        raise ValueError('the simulation year {0} can not be found in file {1}'.format(sim_year, nc_fo))
    
    # get index which corresponds with sim_year in years in nc-file
    sim_year_idx = int(np.where(years == sim_year)[0])
    # get values from data-array for specified year based on index
    nc_arr = nc_var.sel(time=nc_ds.time.values[sim_year_idx])
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('no data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    # open nc-file with rasterio to get affine information
    affine = rio.open(nc_fo).transform

    # initialize output list
    list_out = []
    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(extent_gdf)):
        prov = extent_gdf.iloc[i]
        zonal_stats = rstats.zonal_stats(prov.geometry, nc_arr_vals, affine=affine, stats=stat_func)
        if (zonal_stats[0][stat_func] == None) and (config.getboolean('general', 'verbose')): 
            print('WARNING: NaN computed!')
        list_out.append(zonal_stats[0][stat_func])

    if config.getboolean('general', 'verbose'): print('DEBUG: ... done.')

    return list_out