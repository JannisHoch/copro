import os
import warnings

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterstats as rstats
import xarray as xr
from joblib import Parallel, delayed


def _zonal_one(geom, nc_arr_vals, affine, stat_method, id_val, id_col, warn_on_none):
    """Compute zonal statistics for a single polygon geometry.

    Pulled out so the polygon loop can be dispatched to joblib workers.
    Returns the statistic value, or np.nan if rasterstats returns None.
    """
    zonal_stats = rstats.zonal_stats(
        geom,
        nc_arr_vals,
        affine=affine,
        stats=stat_method,
        all_touched=True,
    )
    val = zonal_stats[0][stat_method]

    if val is None:
        if warn_on_none:
            warnings.warn(
                f"`None` computed for {id_col}{id_val}, setting to `np.nan`!"
            )
        val = np.nan

    return val

def _warn_nan_summary(values, extent_gdf, id_col, var_name, data_year):
    """One warning per variable-year instead of one per polygon.

    data_year is the LAGGED year read from the nc-file, not the simulation
    year: sim_year has already had LAG_TIME subtracted by this point.
    """
    ids = (extent_gdf[id_col].tolist() if id_col in extent_gdf.columns
           else list(range(len(values))))
    nan_ids = [i for i, v in zip(ids, values)
               if v is None or (isinstance(v, float) and np.isnan(v))]
    if not nan_ids:
        return
    shown = ", ".join(str(i) for i in nan_ids[:5])
    more = f", +{len(nan_ids) - 5} more" if len(nan_ids) > 5 else ""
    warnings.warn(f"{var_name} {data_year}: {len(nan_ids)}/{len(values)} "
                  f"polygons NaN ({shown}{more})")

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
    click.echo(f"\tlag time: {LAG_TIME} year(s).")

    if ln_flag:
        raise NotImplementedError(
            "Log-transformation for continuous datetime timestamp not yet implemented."
        )

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

    id_col = config["data"]["extent"].get("id", "watprovID")
    n_jobs = config.get("general", {}).get("n_jobs_zonal", -1)

    # compute zonal stats per polygon in parallel
    # all_touched=True means raster cells touched by polygon are included
    list_out = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_zonal_one)(
            row.geometry,
            nc_arr_vals,
            affine,
            stat_method,
            row[id_col] if id_col in extent_gdf.columns else None,
            id_col,
            True,
        )
        for _, row in extent_gdf.iterrows()
    )
    list_out = list(list_out)

    _warn_nan_summary(list_out, extent_gdf, id_col, var_name, sim_year)

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
    click.echo(f"\tlag time: {LAG_TIME} year(s).")

    if ln_flag:
        raise NotImplementedError(
            "Log-transformation for continuous datetime timestamp not yet implemented."
        )

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

    id_col = config["data"]["extent"].get("id", "watprovID")
    n_jobs = config.get("general", {}).get("n_jobs_zonal", -1)

    # compute zonal stats per polygon in parallel
    # all_touched=True means raster cells touched by polygon are included
    list_out = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_zonal_one)(
            row.geometry,
            nc_arr_vals,
            affine,
            stat_method,
            row[id_col] if id_col in extent_gdf.columns else None,
            id_col,
            True,
        )
        for _, row in extent_gdf.iterrows()
    )
    list_out = list(list_out)

    _warn_nan_summary(list_out, extent_gdf, id_col, var_name, sim_year)

    return list_out
