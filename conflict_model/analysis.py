import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def conflict_in_year_bool(conflict_gdf, extent_gdf, config, sim_year, out_dir, saving_plots=False, showing_plots=False):
    """Determines whether conflict took place in a region in one year and, if so, assigns a value of 1 to this region.

    Arguments:
        conflict_gdf {[type]} -- [description]
        extent_gdf {[type]} -- [description]
        config {[type]} -- [description]
        sim_year {[type]} -- [description]
        out_dir {[type]} -- [description]

    Keyword Arguments:
        saving_plots (bool): whether or not to save the plot (default: False)
        showing_plots (bool): whether or not to show the plot (default: False)

    Returns:
        dataframe: dataframe containing column with boolean information about conflict for each year
    """    
    
    print('determining whether a conflict took place or not')
    
    out_df = extent_gdf.copy()

    # each year initialize new column with default value 0 (=False)
    out_df['boolean_conflict_' + str(sim_year)] = 0
    
    # select the entries which occured in this year
    temp_sel_year = conflict_gdf.loc[conflict_gdf.year == sim_year]   
    
    # merge the dataframes with polygons and conflict information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, out_df)
    
    # determine the aggregated amount of fatalities in one region (e.g. water province)
    fatalities_per_watProv = data_merged['best'].groupby(data_merged['watprovID']).sum().to_frame().rename(columns={"best": 'total_fatalities'})
 
    # loop through all regions and check if exists in sub-set
    # if so, this means that there was conflict and thus assign value 1
    for i in range(len(out_df)):
        i_watProv = out_df.iloc[i]['watprovID']
        if i_watProv in fatalities_per_watProv.index.values:
            fats = int(fatalities_per_watProv.loc[i_watProv])
            out_df.loc[i, 'boolean_conflict_' + str(sim_year)] = 1
    
    print('...DONE' + os.linesep)

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.set_title('boolean_conflict_' + str(sim_year))
    out_df.plot(ax=ax, column='boolean_conflict_' + str(sim_year), legend=True, categorical=True)
    plt.tight_layout()
    
    if saving_plots:
        fn_out = os.path.join(out_dir, 'boolean_conflict_map_' + str(sim_year) + '.png')
        plt.savefig(fn_out, dpi=300)

    if not showing_plots:
        plt.close()

    return out_df