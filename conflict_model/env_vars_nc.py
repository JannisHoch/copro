import xarray as xr
import rasterio as rio
import geopandas as gpd
import rasterstats as rstats
import numpy as np
import matplotlib.pyplot as plt
import os, sys

def rasterstats_GDP_PPP(gdf, extent_gdf, config, sim_year, out_dir, saving_plots=False, showing_plots=False):

    print('calculating zonal statistics per aggregation unit')
    
    nc_fo = os.path.join(config.get('general', 'input_dir'), 
                         config.get('env_vars', 'GDP_PPP'))

    nc_ds = xr.open_dataset(nc_fo)

    nc_var = nc_ds['GDP_per_capita_PPP']

    years = nc_ds['time'].values
    years = years[years>=config.getint('settings', 'y_start')]
    years = years[years<=config.getint('settings', 'y_end')]

    affine = rio.open(nc_fo).transform

    gdf['zonal_stats_min'] = np.nan
    gdf['zonal_stats_max'] = np.nan
    gdf['zonal_stats_mean'] = np.nan

    nc_arr = nc_var.sel(time=sim_year)
    nc_arr_vals = nc_arr.values
    if nc_arr_vals.size == 0:
        raise ValueError('the data was found for this year in the nc-file {}, check if all is correct'.format(nc_fo))

    for i in range(len(gdf)):
        prov = gdf.iloc[i]
        zonal_stats = rstats.zonal_stats(prov.geometry, nc_arr_vals, affine=affine, stats="mean min max")
        gdf.loc[i, 'zonal_stats_min'] = zonal_stats[0]['min']
        gdf.loc[i, 'zonal_stats_max'] = zonal_stats[0]['max']
        gdf.loc[i, 'zonal_stats_mean'] = zonal_stats[0]['mean']

    print('...DONE' + os.linesep)

    fig, axes = plt.subplots(1, 3 , figsize=(20, 10))

    fig.suptitle(str(int(sim_year)), y=0.78)

    gdf.plot(ax=axes[0],
                column='zonal_stats_min',
                vmin=2000,
                vmax=15000,
                legend=True,
                legend_kwds={'label': "min GDP_PPP",
                                'orientation': "vertical",
                                'shrink': 0.5,
                                'extend': 'both'})
    extent_gdf.boundary.plot(ax=axes[0],
                                color='0.5',
                                linestyle=':',
                                label='water province borders')

    gdf.plot(ax=axes[1],
                column='zonal_stats_max',
                vmin=2000,
                vmax=15000,
                legend=True,
                legend_kwds={'label': "max GDP_PPP",
                                'orientation': "vertical",
                                'shrink': 0.5,
                                'extend': 'both'})
    extent_gdf.boundary.plot(ax=axes[1],
                                color='0.5',
                                linestyle=':',
                                label='water province borders')

    gdf.plot(ax=axes[2],
                column='zonal_stats_mean',
                vmin=2000,
                vmax=15000,
                legend=True,
                legend_kwds={'label': "mean GDP_PPP",
                                'orientation': "vertical",
                                'shrink': 0.5,
                                'extend': 'both'})
    extent_gdf.boundary.plot(ax=axes[2],
                                color='0.5',
                                linestyle=':',
                                label='water province borders')

    plt.tight_layout()

    plt_name = 'GDP_PPP_zonal_stats_' + str(int(sim_year)) + '.png'
    plt_name = os.path.join(out_dir, plt_name)

    if saving_plots:
        plt.savefig(plt_name, dpi=300)

    if showing_plots == False:
        plt.close()

    return gdf