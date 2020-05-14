import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def conflict_in_year_bool(conflict_gdf, continent_gdf, config, plotting=False):
    """Determins per year the number of fatalities per country and derivates a boolean value whether conflict has occured in one year in one country or not.

    Arguments:
        conflict_gdf {geodataframe}: geodataframe containing final selection of georeferenced conflicts
        continent_gdf {geodataframe}: geodataframe containing country polygons of selected continent
        config {configuration}: parsed configuration settings

    Keyword Arguments:
        plotting {bool}: whether or not to make annual plots of boolean conflict and conflict fatalities (default: False)
    """    

    # get all years in the dataframe
    years = conflict_gdf.year.unique()

    # go through all years found
    for year in np.sort(years):
        
        # select the entries which occured in this year
        temp_sel_year = conflict_gdf.loc[conflict_gdf.year == year]
        
        # merge this selection with the continent data
        data_merged = gpd.sjoin(temp_sel_year, continent_gdf, how="inner", op='within')
        
        # per country the annual total fatalities are computed and stored in a separate column
        annual_fatalities_sum = pd.merge(continent_gdf,
                                         data_merged['best'].groupby(data_merged['name']).sum().\
                                         to_frame().rename(columns={"best": "best_SUM"}),
                                         on='name')
        
        # if the fatalities exceed 0.0, this entry is assigned a value 1, otherwise 0
        annual_fatalities_sum['conflict_bool'] = np.where(annual_fatalities_sum['best_SUM']>0.0, 1, 0)
            
        # plot results if specified    
        if plotting:
            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10), sharey=True)
    
            annual_fatalities_sum.plot(ax=ax1,column='conflict_bool',
                                           vmin=0,
                                           vmax=2,
                                           categorical=True,
                                           legend=True)

            continent_gdf.boundary.plot(ax=ax1,
                                        color='0.5',
                                        linestyle=':')

            ax1.set_xlim(continent_gdf.total_bounds[0]-1, continent_gdf.total_bounds[2]+1)
            ax1.set_ylim(continent_gdf.total_bounds[1]-1, continent_gdf.total_bounds[3]+1)
            ax1.set_title('conflict_bool ' + str(year))
            
            annual_fatalities_sum.plot(ax=ax2, column='best_SUM',
                                           vmin=0,
                                           vmax=1500)

            continent_gdf.boundary.plot(ax=ax2,
                                        color='0.5',
                                        linestyle=':')

            ax2.set_xlim(continent_gdf.total_bounds[0]-1, continent_gdf.total_bounds[2]+1)
            ax2.set_ylim(continent_gdf.total_bounds[1]-1, continent_gdf.total_bounds[3]+1)
            ax2.set_title('aggr. fatalities ' + str(year))

    return 
