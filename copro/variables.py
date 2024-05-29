import xarray as xr
import rasterio as rio
import pandas as pd
import geopandas as gpd
import rasterstats as rstats
import numpy as np
import os
import math
import click

import warnings

warnings.filterwarnings("once")


def nc_with_float_timestamp(
    extent_gdf: gpd.GeoDataFrame,
    config: dict,
    root_dir: str,
    var_name: str,
    sim_year: int,
) -> list:
    """This function extracts a value from a netCDF-file (specified in the yaml-file)
    for each polygon specified in extent_gdf for a given year.
    In the yaml-file, it must also be specified whether the value is log-transformed or not,
    and which statistical method is applied.

    .. note::
        The key in the yaml-file must be identical to variable name in netCDF-file.

    .. note::
        Works only with nc-files with annual data.

    Args:
        extent_gdf (gpd.GeoDataFrame): One or more polygons with geometry information for which values are extracted.
        config (dict): Parsed configuration settings of run.
        root_dir (str): Path to location of yaml-file.
        var_name (str): Name of variable in nc-file. Must be the same as is specified in yaml-file.
        sim_year (int): Year for which data is extracted.

    Returns:
        list: List containing statistical value per polygon, i.e. with same length as extent_gdf.
    """

    nc_fo = os.path.join(
        root_dir,
        config["general"]["input_dir"],
        config["data"]["indicators"][var_name]["file"],
    )

    if "log" not in config["data"]["indicators"][var_name].keys():
        ln_flag = False
    else:
        ln_flag = config["data"]["indicators"][var_name]["log"]
    if "stat" not in config["data"]["indicators"][var_name].keys():
        stat_method = "mean"
    else:
        stat_method = config["data"]["indicators"][var_name]["stat"]
    LAG_TIME = 1
    click.echo(f"\tuse log: {ln_flag}.")
    click.echo(f"\tstatistical method: {stat_method}.")
    click.echo(f"\tLAG TIME: {LAG_TIME} year(s).")

    sim_year = sim_year - LAG_TIME

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]
    # open nc-file with rasterio to get affine information
    affine = rio.open(nc_fo).transform

    # get values from data-array for specified year
    nc_arr_vals = nc_var.sel({"time": sim_year}).values
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
    config: dict,
    root_dir: str,
    var_name: str,
    sim_year: int,
) -> list:
    """This function extracts a value from a netCDF-file (specified in the yaml-file)
    for each polygon specified in extent_gdf for a given year.
    In the yaml-file, it must also be specified whether the value is log-transformed or not,
    and which statistical method is applied.

    .. note::
        The key in the yaml-file must be identical to variable name in netCDF-file.

    .. note::
        Works only with nc-files with annual data.

    Args:
        extent_gdf (gpd.GeoDataFrame): One or more polygons with geometry information for which values are extracted.
        config (config): Parsed configuration settings of run.
        root_dir (str): Path to location of yaml-file.
        var_name (str): Name of variable in nc-file. Must be the same as in the yaml-file.
        sim_year (int): Year for which data is extracted.

    Returns:
        list: List containing statistical value per polygon, i.e. with same length as extent_gdf.
    """

    nc_fo = os.path.join(
        root_dir,
        config["general"]["input_dir"],
        config["data"]["indicators"][var_name]["file"],
    )

    if "log" not in config["data"]["indicators"][var_name].keys():
        ln_flag = False
    else:
        ln_flag = config["data"]["indicators"][var_name]["log"]
    if "stat" not in config["data"]["indicators"][var_name].keys():
        stat_method = "mean"
    else:
        stat_method = config["data"]["indicators"][var_name]["stat"]
    LAG_TIME = 1
    click.echo(f"\tuse log: {ln_flag}.")
    click.echo(f"\tstatistical method: {stat_method}.")
    click.echo(f"\tLAG TIME: {LAG_TIME} year(s).")

    sim_year = sim_year - LAG_TIME
    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]

    # get values from data-array for specified year based on index
    nc_arr_vals = nc_var.sel({"time": pd.to_datetime(sim_year, format="%Y")}).values
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
