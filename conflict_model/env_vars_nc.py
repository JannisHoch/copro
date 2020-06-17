import xarray as xr
import rasterio as rio
import pandas as pd
import geopandas as gpd
import rasterstats as rstats
import numpy as np
import matplotlib.pyplot as plt
import os, sys

def rasterstats_GDP_PPP(gdf, config, sim_year, out_dir, saving_plots=False, showing_plots=False):

    print('calculating GDP PPP mean per aggregation unit')
    
    nc_fo = os.path.join(config.get('general', 'input_dir'), 
                         config.get('env_vars', 'GDP_PPP'))

    nc_ds = xr.open_dataset(nc_fo)

    nc_var = nc_ds['GDP_per_capita_PPP']

    # years = pd.to_datetime(nc_ds.time.values).to_period(freq='Y').strftime('%Y').to_numpy(dtype=int)
    # if sim_year not in years:
    #     raise ValueError('the simulation year {0} can not be found in file {1}'.format(sim_year, nc_fo))
    # sim_year_idx = int(np.where(years == sim_year)[0])

    affine = rio.open(nc_fo).transform

    # gdf['zonal_stats_min_' + str(sim_year)] = np.nan
    # gdf['zonal_stats_max_' + str(sim_year)] = np.nan
    # gdf['GDP_PPP_mean_' + str(sim_year)] = np.nan

    nc_arr = nc_var.sel(time=sim_year)
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('the data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    list_GDP_PPP = []
    
    for i in range(len(gdf)):
        prov = gdf.iloc[i]
        zonal_stats = rstats.zonal_stats(prov.geometry, nc_arr_vals, affine=affine, stats="mean")
        # gdf.loc[i, 'zonal_stats_min_' + str(sim_year)] = zonal_stats[0]['min']
        # gdf.loc[i, 'zonal_stats_max_' + str(sim_year)] = zonal_stats[0]['max']
        list_GDP_PPP.append(zonal_stats[0]['mean'])

    print('...DONE' + os.linesep)

    return list_GDP_PPP

def rasterstats_totalEvap(gdf_in, config, sim_year, out_dir):

    print('calculating evaporation mean per aggregation unit')
    
    nc_fo = os.path.join(config.get('general', 'input_dir'), 
                         config.get('env_vars', 'evaporation'))

    nc_ds = xr.open_dataset(nc_fo)

    nc_var = nc_ds['total_evaporation']

    years = nc_ds['time'].values
    years = years[years>=config.getint('settings', 'y_start')]
    years = years[years<=config.getint('settings', 'y_end')]

    affine = rio.open(nc_fo).transform

    gdf = gdf_in.copy()

    gdf['evap_mean_' + str(sim_year)] = np.nan

    nc_arr = nc_var.sel(time=sim_year)
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('the data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    for i in range(len(gdf)):
        prov = gdf.iloc[i]
        zonal_stats = rstats.zonal_stats(prov.geometry, nc_arr_vals, affine=affine, stats="mean")
        gdf.loc[i, 'evap_mean_' + str(sim_year)] = zonal_stats[0]['mean']

    print('...DONE' + os.linesep)

    return gdf