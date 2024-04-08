import xarray as xr
import rasterio as rio
import pandas as pd
import geopandas as gpd
import rasterstats as rstats
import numpy as np
import os
import math
import click
from distutils import util
from configparser import RawConfigParser

import warnings

warnings.filterwarnings("once")


def nc_with_float_timestamp(
    extent_gdf: gpd.GeoDataFrame,
    config: RawConfigParser,
    root_dir: str,
    var_name: str,
    sim_year: int,
) -> list:
    """This function extracts a value from a netCDF-file (specified in the cfg-file)
    for each polygon specified in extent_gdf for a given year.
    In the cfg-file, it must also be specified whether the value is log-transformed or not,
    and which statistical method is applied.

    .. note::
        The key in the cfg-file must be identical to variable name in netCDF-file.

    .. note::
        Works only with nc-files with annual data.

    Args:
        extent_gdf (gpd.GeoDataFrame): One or more polygons with geometry information for which values are extracted.
        config (RawConfigParser): parsed configuration settings of run.
        root_dir (str): path to location of cfg-file.
        var_name (str): name of variable in nc-file. Must be the same as is specified in cfg-file.
        sim_year (int): year for which data is extracted.

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as extent_gdf.
    """

    # get the filename, True/False whether log-transform shall be applied, and statistical method from cfg-file as list
    data_fo = os.path.join(
        root_dir, config.get("general", "input_dir"), config.get("data", var_name)
    ).rsplit(",")

    # if not all of these three aspects are provided, raise error
    if len(data_fo) != 3:
        raise ValueError(
            "Not all settings for input data set {} provided - \
                it must contain of path, False/True, and statistical method".format(
                os.path.join(
                    root_dir,
                    config.get("general", "input_dir"),
                    config.get("data", var_name),
                )
            )
        )

    # if not, split the list into separate variables
    nc_fo = data_fo[0]
    ln_flag = bool(util.strtobool(data_fo[1]))
    stat_method = str(data_fo[2])

    lag_time = 1
    click.echo(f"Applying {lag_time} year lag time.")
    sim_year = sim_year - lag_time

    if ln_flag:
        click.echo(
            "Calculating log-transformed {0} {1} per aggregation unit from file {2} for year {3}".format(
                stat_method, var_name, nc_fo, sim_year
            )
        )
    else:
        click.echo(
            "Calculating {0} {1} per aggregation unit from file {2} for year {3}".format(
                stat_method, var_name, nc_fo, sim_year
            )
        )

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]
    if ln_flag:
        nc_var = np.log(nc_var)
    # open nc-file with rasterio to get affine information
    affine = rio.open(nc_fo).transform

    # get values from data-array for specified year
    nc_arr = nc_var.sel(time=sim_year)
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError(
            f"No data was found for this year in the nc-file {nc_fo}, check if all is correct."
        )

    # initialize output list
    list_out = []
    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(extent_gdf)):

        # province i
        prov = extent_gdf.iloc[i]

        # compute zonal stats for this province
        # computes a value per polygon for all raster cells that are touched by polygon (all_touched=True)
        # if all_touched=False, only for raster cells with centre point in polygon are considered,
        # but this is problematic for very small polygons
        zonal_stats = rstats.zonal_stats(
            prov.geometry,
            nc_arr_vals,
            affine=affine,
            stats=stat_method,
            all_touched=True,
        )
        val = zonal_stats[0][stat_method]

        # # if specified, log-transform value
        if ln_flag:
            # works only if zonal stats is not None, i.e. if it's None it stays None
            val_ln = np.log(val)
            # in case log-transformed value results in -inf, replace with None
            if val_ln == -math.inf:
                val = np.log(val + 1)
            else:
                val = val_ln

        # warn if result is NaN
        if val is math.nan:
            warnings.warn("NaN computed!")

        list_out.append(val)

    return list_out


def nc_with_continous_datetime_timestamp(
    extent_gdf: gpd.GeoDataFrame,
    config: RawConfigParser,
    root_dir: str,
    var_name: str,
    sim_year: int,
) -> list:
    """This function extracts a value from a netCDF-file (specified in the cfg-file)
    for each polygon specified in extent_gdf for a given year.
    In the cfg-file, it must also be specified whether the value is log-transformed or not,
    and which statistical method is applied.

    .. note::
        The key in the cfg-file must be identical to variable name in netCDF-file.

    .. note::
        Works only with nc-files with annual data.

    Args:
        extent_gdf (gpd.GeoDataFrame): One or more polygons with geometry information for which values are extracted.
        config (RawConfigParser): parsed configuration settings of run.
        root_dir (str): path to location of cfg-file.
        var_name (str): name of variable in nc-file. Must be the same as in the cfg-file.
        sim_year (int): year for which data is extracted.

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as extent_gdf.
    """

    # get the filename, True/False whether log-transform shall be applied, and statistical method from cfg-file as list
    data_fo = os.path.join(
        root_dir, config.get("general", "input_dir"), config.get("data", var_name)
    ).rsplit(",")

    # if not all of these three aspects are provided, raise error
    if len(data_fo) != 3:
        raise ValueError(
            "Not all settings for input data set {} provided - \
                it must contain of path, False/True, and statistical method".format(
                os.path.join(
                    root_dir,
                    config.get("general", "input_dir"),
                    config.get("data", var_name),
                )
            )
        )

    # if not, split the list into separate variables
    nc_fo = data_fo[0]
    ln_flag = bool(util.strtobool(data_fo[1]))
    stat_method = str(data_fo[2])

    lag_time = 1
    click.echo(f"Applying {lag_time} year lag time for variable {var_name}.")
    sim_year = sim_year - lag_time

    if ln_flag:
        click.echo(
            "Calculating log-transformed {0} {1} per aggregation unit from file {2} for year {3}".format(
                stat_method, var_name, nc_fo, sim_year
            )
        )
    else:
        click.echo(
            "Calculating {0} {1} per aggregation unit from file {2} for year {3}".format(
                stat_method, var_name, nc_fo, sim_year
            )
        )

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]
    # get years contained in nc-file as integer array to be compatible with sim_year
    years = (
        pd.to_datetime(nc_ds.time.values)
        .to_period(freq="Y")
        .strftime("%Y")
        .to_numpy(dtype=int)
    )
    if sim_year not in years:
        warnings.warn(
            f"The simulation year {sim_year} can not be found in file {nc_fo}."
        )
        warnings.warn(
            "Using the next following year instead (yes that is an ugly solution...)"
        )
        sim_year = sim_year + 1
        # raise ValueError('ERROR: the simulation year {0} can not be found in file {1}'.format(sim_year, nc_fo))

    # get index which corresponds with sim_year in years in nc-file
    sim_year_idx = int(np.where(years == sim_year)[0])
    # get values from data-array for specified year based on index
    nc_arr = nc_var.sel(time=nc_ds.time.values[sim_year_idx])
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError(
            "No data was found for this year in the nc-file {}, check if all is correct".format(
                nc_fo
            )
        )

    # open nc-file with rasterio to get affine information
    affine = rio.open(nc_fo).transform

    # initialize output list
    list_out = []
    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(extent_gdf)):

        # province i
        prov = extent_gdf.iloc[i]

        # compute zonal stats for this province
        # computes a value per polygon for all raster cells that are touched by polygon (all_touched=True)
        # if all_touched=False, only for raster cells with centre point in polygon are considered,
        # but this is problematic for very small polygons
        zonal_stats = rstats.zonal_stats(
            prov.geometry,
            nc_arr_vals,
            affine=affine,
            stats=stat_method,
            all_touched=True,
        )
        val = zonal_stats[0][stat_method]

        # # if specified, log-transform value
        if ln_flag:
            # works only if zonal stats is not None, i.e. if it's None it stays None
            val_ln = np.log(val)
            # in case log-transformed value results in -inf, replace with None
            if val_ln == -math.inf:
                val = np.log(val + 1)
            else:
                val = val_ln

        # warn if result is NaN
        if val is math.nan:
            warnings.warn("NaN computed!")

        list_out.append(val)

    return list_out
