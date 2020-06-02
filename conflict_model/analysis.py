import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def conflict_in_year_bool(conflict_gdf, extent_gdf, config, sim_year, out_dir, saving_plots=False, showing_plots=False):
    """Determins per year the number of fatalities per country and derivates a boolean value whether conflict has occured in one year in one country or not.

    Arguments:
        conflict_gdf {geodataframe}: geodataframe containing final selection of georeferenced conflicts
        extent_gdf {geodataframe}: geodataframe containing country polygons of selected extent
        config {configuration}: parsed configuration settings

    Keyword Arguments:
        plotting {bool}: whether or not to make annual plots of boolean conflict and conflict fatalities (default: False)
    """  
        
    print('determining whether a conflict took place or not...')
    
    # select the conflict entries which occured in this year
    conflict_gdf_perYear = conflict_gdf.loc[conflict_gdf.year == sim_year]
    
    # merge this selection with the continent data
    # this adds the water province where conflict took place to each selected conflict entry
    extent_conflict_merged = gpd.sjoin(conflict_gdf_perYear, extent_gdf, how="inner", op='within')

    # sum fatalities per water province and create dataframe with new column name; index is the water province ID
    fatalities_per_waterProvince = extent_conflict_merged['best'].groupby(extent_conflict_merged['watprovID']).sum().to_frame().rename(columns={"best": "best_SUM"})
    
    # per water province the annual total fatalities are computed and stored in a separate column
    # this dataframe may be smaller than extent_conflict_merged as multiple conflicts may be in one water province
    extent_waterProvinces_with_boolFatalities = pd.merge(extent_gdf,
                                                     fatalities_per_waterProvince,
                                                     on='watprovID')
    
    # if the fatalities exceed 0.0, this entry is assigned a value 1, otherwise 0
    extent_waterProvinces_with_boolFatalities['conflict_bool'] = np.where(extent_waterProvinces_with_boolFatalities['best_SUM']>0.0, 1, 0)

    print('...DONE' + os.linesep)
        
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10), sharey=True)

    extent_waterProvinces_with_boolFatalities.plot(ax=ax1,
                                               column='conflict_bool',
                                               vmin=0,
                                               vmax=2,
                                               categorical=True,
                                               legend=True)

    conflict_gdf_perYear.plot(ax=ax1, 
                              legend=True, 
                              color='r', 
                              label='PRIO/UCDP events')

    extent_gdf.boundary.plot(ax=ax1,
                             color='0.5',
                             linestyle=':',
                             label='water province borders')

    ax1.set_xlim(extent_gdf.total_bounds[0]-1, extent_gdf.total_bounds[2]+1)
    ax1.set_ylim(extent_gdf.total_bounds[1]-1, extent_gdf.total_bounds[3]+1)
    ax1.set_title('conflict_bool ' + str(sim_year))

    ax1.legend()
    
    extent_waterProvinces_with_boolFatalities.plot(ax=ax2, 
                                               column='best_SUM',
                                               vmin=0,
                                               vmax=1500,
                                               legend=True,
                                               legend_kwds={'label': "FATALITIES_SUM",
                                                            'orientation': "vertical"},)

    extent_gdf.boundary.plot(ax=ax2,
                             color='0.5',
                             linestyle=':')

    ax2.set_xlim(extent_gdf.total_bounds[0]-1, extent_gdf.total_bounds[2]+1)
    ax2.set_ylim(extent_gdf.total_bounds[1]-1, extent_gdf.total_bounds[3]+1)
    ax2.set_title('aggr. fatalities ' + str(sim_year))

    fn_out = os.path.join(out_dir, 'boolean_conflict_map_' + str(sim_year) + '.png')
    
    if saving_plots:
        plt.savefig(fn_out, dpi=300)

    if not showing_plots:
        plt.close()

    return conflict_gdf_perYear, extent_conflict_merged, fatalities_per_waterProvince, extent_waterProvinces_with_boolFatalities
