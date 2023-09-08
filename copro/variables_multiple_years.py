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

import warnings
warnings.filterwarnings("ignore")

def nc_with_float_timestamp(migration_gdf, config, root_dir, var_name, sim_year):
    """This function extracts a value from a netCDF-file (specified in the cfg-file) for each unique polygon specified in migration_gdf for a given year.
    In the cfg-file, it must also be specified whether the value is log-transformed or not, and which statistical method is applied.

    NOTE:
    The key in the cfg-file must be identical to variable name in netCDF-file. 

    NOTE:
    This function is specifically written for netCDF-files where the time variable contains integer (year-)values, e.g. 1995, 1996, ...

    NOTE:
    Works only with nc-files with annual data.

    Args:
        migration_gdf (geo-dataframe): geo-dataframe containing migration, polygon-geometry and polygon-ID information
        config (config): parsed configuration settings of run.
        root_dir (str): path to location of cfg-file. 
        var_name (str): name of variable in nc-file, must also be the same under which path to nc-file is specified in cfg-file.
        sim_year (int): year for which data is extracted.

    Raises:
        ValueError: raised if not everything is specified in cfg-file.
        ValueError: raised if the extracted variable at a time step does not contain data.

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as number of unique polygons in migration_gdf.
    """   

    # get the filename, True/False whether log-transform shall be applied, and statistical method from cfg-file as list
    data_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name)).rsplit(',')

    # if not all of these three aspects are provided, raise error
    if len(data_fo) != 3:
            raise ValueError('ERROR: not all settings for input data set {} provided - it must contain of path, False/True, and statistical method'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name))))
    
    # if not, split the list into separate variables
    else:
            nc_fo = data_fo[0]
            ln_flag = bool(util.strtobool(data_fo[1]))
            stat_method = str(data_fo[2])

    if config.getboolean('timelag', var_name): 
            lag_time = 1
            click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))
    else:
            lag_time =0
            click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))

    # if config.getboolean('general', 'verbose'): click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))
    sim_year = sim_year - lag_time

    # get years_to_average depending on the config settings
    if config.getboolean('general', 'three_year_migration_average'):
        years_to_average = [sim_year, sim_year + 1, sim_year + 2]                       
    
    elif config.getboolean('general', 'five_year_migration_average'):
        years_to_average = [sim_year, sim_year + 1, sim_year + 2, sim_year + 3, sim_year + 4]

    else:
        raise ValueError('ERROR: please cheack cfg file regarding averaging years')

    if config.getboolean('general', 'verbose'): 
        if ln_flag:
            click.echo('DEBUG: calculating log-transformed {0} {1} per aggregation unit from file {2} for years {3}'.format(stat_method, var_name, nc_fo, years_to_average))
        else:
            click.echo('DEBUG: calculating average {0} {1} per aggregation unit from file {2} for years {3}'.format(stat_method, var_name, nc_fo, years_to_average))

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)

     # If lon and lat are spatial dimensions, rename those to x and y to avoid errors
    if 'lon' in nc_ds.dims:
        nc_ds = nc_ds.rename({'lon': 'x'})
    if 'lat' in nc_ds.dims:
        nc_ds = nc_ds.rename({'lat': 'y'})

    # if y-axis is flipped, flip it back
    if nc_ds.rio.transform().e > 0:
        nc_ds = nc_ds.reindex(y=list(reversed(nc_ds.y)))

    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]

    # Ensure the variable values are in float32
    if nc_var.values.dtype != np.float32:
        nc_var = nc_var.astype(np.float32)

    # Get the non-zero and positive mask along the time dimension
    non_zero_positive_mask = nc_var.values > 0

    # Apply logarithm only to positive values using the mask along the time dimension
    nc_var.values = xr.where(non_zero_positive_mask, np.log(nc_var.values), nc_var)

    # Handle cases where log-transformed value results in -inf
    nc_var = xr.where(nc_var.values == -math.inf, 0, nc_var)

    # open nc-file with rasterio to get affine information
    affine = nc_ds.rio.transform()

    # get values from data-array for specified years
    nc_arr_vals = nc_var.sel(time=years_to_average)

    # Handle cases where nc_arr_vals is empty (i.e., no data found for the specified year)
    if nc_arr_vals.size == 0:
        print('WARNING: No data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))
        return []

    # load crs from config file, if not specified, get crs from nc-file. If neither is specified, raise error
    crs = config.get('crs', var_name) or nc_var.rio.crs
    assert crs is not None, 'ERROR: no CRS found for variable {}'.format(var_name)

    # convert migration_gdf to crs of nc-file
    migration_gdf_crs_corrected = migration_gdf.to_crs(crs)
    
    # Extract the data values from the xarray DataArray
    nc_arr_vals_data = nc_arr_vals.values

    # initialize output list and a set to keep track of processed polygons
    list_out = []
    processed_polygons = set()

    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(migration_gdf_crs_corrected)):

        # province i
        polygon = migration_gdf_crs_corrected.iloc[i]

        # Check if the polygon has already been processed, if yes, skip to the next iteration
        if polygon.GID_2 in processed_polygons:
            continue

        # Mark the current polygon as processed
        processed_polygons.add(polygon.GID_2)
        
        avg_values = []
        
        # compute zonal stats for this polygon
        # computes a value per polygon for all raster cells that are touched by polygon (all_touched=True)
        # if all_touched=False, only for raster cells with centre point in polygon are considered, but this is problematic for very small polygons
        #zonal_stats = rstats.zonal_stats(polygon.geometry, nc_arr_vals, affine=affine, stats=stat_method, all_touched=True, nodata=np.nan)

        #val = zonal_stats[0][stat_method]

        # Loop through years to average (sim_year, sim_year + 1, sim_year + 2)
        for year in years_to_average:
            time_values = nc_ds.time.values
            year_idx = np.where(time_values == year)[0][0]
            nc_arr = nc_var.sel(time=nc_ds.time.values[year_idx])
            nc_arr_vals = nc_arr.values

            if nc_arr_vals.size == 0:
                raise ValueError('ERROR: no data was found for year {} in the nc-file {}, check if all is correct'.format(year, nc_fo))

            zonal_stats = rstats.zonal_stats(polygon.geometry, nc_arr_vals, affine=affine, stats=stat_method, all_touched=True, nodata=np.nan)
            if not zonal_stats:
                print("No valid statistics found for polygon:", polygon.GID_2)
                val = 0  # or np.nan
            else:
                val = zonal_stats[0][stat_method]
                # compute zonal stats for this polygon and this year 
                zonal_stats_result = rstats.zonal_stats(polygon.geometry, nc_arr_vals, affine=affine, stats=stat_method, all_touched=True, nodata=np.nan)
                val = zonal_stats_result[0][stat_method]

                # Apply log-transform if needed
                if ln_flag:
                    if val is not None:
                        val = np.log(val)
                    else:
                        click.echo('WARNING: a value of {} for ID {} was computed - no good!'.format(np.log(val+1), polygon.GID_2))

            avg_values.append(val)  # Append the computed value for this year to avg_values

        # Calculate the average value over the three years for this polygon
        average_value = np.mean(avg_values)

        # Append the tuple (average value, GID_2) to list_out
        list_out.append((average_value, polygon.GID_2))

    return list_out

def nc_with_continous_datetime_timestamp(migration_gdf, config, root_dir, var_name, sim_year):
    """This function extracts a value from a netCDF-file (specified in the cfg-file) for each unique polygon specified in migration_gdf for a given year.
    In the cfg-file, it must also be specified whether the value is log-transformed or not, and which statistical method is applied.

    NOTE:
    The key in the cfg-file must be identical to variable name in netCDF-file. 

    NOTE:
    Works only with nc-files with annual data.

    Args:
        migration_gdf (geo-dataframe): geo-dataframe containing migration, polygon-geometry and polygon-ID information
        config (config): parsed configuration settings of run.
        root_dir (str): path to location of cfg-file. 
        var_name (str): name of variable in nc-file, must also be the same under which path to nc-file is specified in cfg-file.
        sim_year (int): year for which data is extracted.

    Raises:
        ValueError: raised if not everything is specified in cfg-file.
        ValueError: raised if specfied year cannot be found in years in nc-file.
        ValueError: raised if the extracted variable at a time step does not contain data.

    Returns:
         list: list of tuples, where each tuple contains the computed value and its corresponding GID_2 ID.
    """   

    # get the filename, True/False whether log-transform shall be applied, and statistical method from cfg-file as list
    data_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name)).rsplit(',')

    # if not all of these three aspects are provided, raise error
    if len(data_fo) != 3:
        raise ValueError('ERROR: not all settings for input data set {} provided - it must contain of path, False/True, and statistical method'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name))))
    
    # if not, split the list into separate variables
    else:
        nc_fo = data_fo[0]
        ln_flag = bool(util.strtobool(data_fo[1]))
        stat_method = str(data_fo[2])

    if config.getboolean('timelag', var_name): 
            lag_time = 1
            click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))
    else:
            lag_time =0
            click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))
    
    #if config.getboolean('general', 'verbose'): click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))
    sim_year = sim_year - lag_time
  
    # get years_to_average depending on the config settings
    if config.getboolean('general', 'three_year_migration_average'):
        years_to_average = [sim_year, sim_year + 1, sim_year + 2]                       
    
    elif config.getboolean('general', 'five_year_migration_average'):
        years_to_average = [sim_year, sim_year + 1, sim_year + 2, sim_year + 3, sim_year + 4]

    else:
        raise ValueError('ERROR: please cheack cfg file regarding averaging years')

    if config.getboolean('general', 'verbose'): 
        if ln_flag:
            click.echo('DEBUG: calculating log-transformed {0} {1} per aggregation unit from file {2} for year {3}'.format(stat_method, var_name, nc_fo, years_to_average))
        else:
            click.echo('DEBUG: calculating {0} {1} per aggregation unit from file {2} for year {3}'.format(stat_method, var_name, nc_fo, years_to_average))

    # open nc-file with xarray as dataset
    nc_ds = xr.open_dataset(nc_fo)
    
    # If lon and lat are spatial dimensions, rename those to x and y to avoid errors
    if 'lon' in nc_ds.dims:
        nc_ds = nc_ds.rename({'lon': 'x'})
    if 'lat' in nc_ds.dims:
        nc_ds = nc_ds.rename({'lat': 'y'})

    # if y-axis is flipped, flip it back
    if nc_ds.rio.transform().e > 0:
        nc_ds = nc_ds.reindex(y=list(reversed(nc_ds.y)))

    # get xarray data-array for specified variable
    nc_var = nc_ds[var_name]
    if nc_var.values.dtype != np.float32:
        nc_var = nc_var.astype(np.float32)

    # get years contained in nc-file as integer array to be compatible with sim_year
    years = pd.to_datetime(nc_ds.time.values).to_period(freq='Y').strftime('%Y').to_numpy(dtype=int)
    if sim_year not in years:
        click.echo('WARNING: the simulation year {0} can not be found in file {1}'.format(sim_year, nc_fo))
        click.echo('WARNING: using the next following year instead (yes that is an ugly solution...)')
        sim_year = sim_year + 1
    
    # get index which corresponds with sim_year in years in nc-file
    sim_year_idx = int(np.where(years == sim_year)[0])
    # get values from data-array for specified year based on index
    nc_arr = nc_var.sel(time=nc_ds.time.values[sim_year_idx])
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('ERROR: no data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    # open nc-file with rasterio to get affine information
    # get affine from xarray data-array
    affine = nc_ds.rio.transform()

    # load crs from config file, if not specified, get crs from nc-file. If neither is specified, raise error
    crs = config.get('crs', var_name) or nc_var.rio.crs
    assert crs is not None, 'ERROR: no CRS found for variable {}'.format(var_name)

    # convert migration_gdf to crs of nc-file
    migration_gdf_crs_corrected = migration_gdf.to_crs(crs)

    # initialize output list and a set to keep track of processed polygons
    list_out = []
    processed_polygons = set()

    # loop through all polygons in geo-dataframe and compute statistics, then append to output file
    for i in range(len(migration_gdf_crs_corrected)):

        # province i
        polygon = migration_gdf_crs_corrected.iloc[i]

        # Check if the polygon has already been processed, if yes, skip to the next iteration
        if polygon.GID_2 in processed_polygons:
            continue

        # Mark the current polygon as processed
        processed_polygons.add(polygon.GID_2)
        
        avg_values = []

        # Loop through years to average (sim_year, sim_year + 1, sim_year + 2)
        for year in years_to_average:
            year_idx = int(np.where(years == year)[0])
            nc_arr = nc_var.sel(time=nc_ds.time.values[year_idx])
            nc_arr_vals = nc_arr.values

            if nc_arr_vals.size == 0:
                raise ValueError('ERROR: no data was found for year {} in the nc-file {}, check if all is correct'.format(year, nc_fo))

            zonal_stats = rstats.zonal_stats(polygon.geometry, nc_arr_vals, affine=affine, stats=stat_method, all_touched=True, nodata=np.nan)
            if not zonal_stats:
                print("No valid statistics found for polygon:", polygon.GID_2)
                val = 0  # or np.nan
            else:
                val = zonal_stats[0][stat_method]
                # compute zonal stats for this polygon and this year 
                zonal_stats_result = rstats.zonal_stats(polygon.geometry, nc_arr_vals, affine=affine, stats=stat_method, all_touched=True, nodata=np.nan)
                val = zonal_stats_result[0][stat_method]

                # Apply log-transform if needed
                if ln_flag:
                    if val is not None:
                        val = np.log(val)
                    else:
                        click.echo('WARNING: a value of {} for ID {} was computed - no good!'.format(np.log(val+1), polygon.GID_2))

            avg_values.append(val)  # Append the computed value for this year to avg_values

        # Calculate the average value over the three years for this polygon
        average_value = np.mean(avg_values)

        # Append the tuple (average value, GID_2) to list_out
        list_out.append((average_value, polygon.GID_2))

    return list_out

def csv_extract_value(migration_gdf, config, root_dir, var_name, sim_year):
    """This function extracts an average value from a csv-file (specified in the cfg-file) for each polygon specified in migration_gdf for sim_year the two following years.
    In the cfg-file, it must also be specified whether the value is log-transformed or not, and which statistical method is applied.

    NOTE:
    The key in the cfg-file must be identical to variable name in csv-file. 

    NOTE:
    Works only with csv-files with annual data.

    Args:
        migration_gdf (geo-dataframe): geo-dataframe containing migration, polygon-geometry and polygon-ID information
        config (config): parsed configuration settings of run.
        root_dir (str): path to location of cfg-file. 
        var_name (str): name of variable in file, must also be the same under which path to csv-file is specified in cfg-file.
        sim_year (int): year for which data is extracted.

    Raises:
        ValueError: raised if not everything is specified in cfg-file.
        ValueError: raised if specfied year cannot be found in years in nc-file.
        ValueError: raised if the extracted variable at a time step does not contain data.

    Returns:
        list: list containing statistical value per polygon, i.e. with same length as number of unique polygons in migration_gdf.
    """   

    # get the filename, True/False whether log-transform shall be applied, and statistical method from cfg-file as list
    data_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name)).rsplit(',')

    # if not all of these three aspects are provided, raise error
    if len(data_fo) != 3:
        raise ValueError('ERROR: not all settings for input data set {} provided - it must contain of path, False/True, and statistical method'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', var_name))))
    
    # if not, split the list into separate variables
    else:
        csv_fo = data_fo[0] 
        ln_flag = bool(util.strtobool(data_fo[1]))
        stat_method = str(data_fo[2]) # not needed, since the value per polygon is already given in the csv

    if config.getboolean('timelag', var_name): 
            lag_time = 1
            click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))
    else:
            lag_time =0
            click.echo('DEBUG: applying {} year lag time for variable {}'.format(lag_time, var_name))
    
    sim_year = sim_year - lag_time

    # Read the CSV file
    csv_data = pd.read_csv(csv_fo)

    # Check if the specified year is in the available years, and if not, find the nearest year.
    available_years = csv_data['year'].unique()
    if sim_year not in available_years:
        nearest_year = min(available_years, key=lambda x: abs(x - sim_year))
        click.echo(f'WARNING: Year {sim_year} not found in the CSV data. Using nearest year: {nearest_year}')
        best_year = nearest_year
        click.echo('INFO: using {} year lag time for variable {}'.format(nearest_year, var_name))
    else:
        best_year = sim_year

    # get years_to_average depending on the config settings
    if config.getboolean('general', 'three_year_migration_average'):
        years_to_average = [best_year, best_year + 1, best_year + 2]                       
    
    elif config.getboolean('general', 'five_year_migration_average'):
        years_to_average = [best_year, best_year + 1, best_year + 2, best_year + 3, best_year + 4]

    else:
        raise ValueError('ERROR: please cheack cfg file regarding averaging years')

    # select the polygons that must be selected
    polygon_names = migration_gdf['GID_2'].unique().tolist()

    all_selected_data = []

    for year in years_to_average:
        selected_csv_data = csv_data[csv_data['GID_2'].isin(polygon_names) & (csv_data['year'] == year)]

        if selected_csv_data.empty:
            raise ValueError(f'ERROR: No data was found for year {year} in the CSV file {var_name}, check if all is correct')

        all_selected_data.append(selected_csv_data)

    avg_values = {}
    for selected_csv_data in all_selected_data:
        for _, row in selected_csv_data.iterrows():
            value = row[var_name]
            GID_2 = row['GID_2']

            if ln_flag:  # Apply log transformation if needed
                value = np.log(value)

            if GID_2 not in avg_values:
                avg_values[GID_2] = []
            avg_values[GID_2].append(value)

    # Create a list of tuples with average value and GID_2 ID
    list_out = [(np.mean(values_list), GID_2) for GID_2, values_list in avg_values.items()]

    return list_out