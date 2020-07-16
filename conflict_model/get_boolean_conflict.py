import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def conflict_in_year_bool(conflict_gdf, extent_gdf, config, sim_year): 
    """Creates a list for each timestep with boolean information whether a conflict took place in a polygon or not.

    Args:
        conflict_gdf (geodataframe): geo-dataframe containing georeferenced information of conflict (tested with PRIO/UCDP data)
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted
        config (config): parsed configuration settings of run
        sim_year (int): year for which data is extracted

    Raises:
        AssertionError: raised if the length of output list does not match length of input geo-dataframe

    Returns:
        list: list containing 0/1 per polygon depending on conflict occurence
    """    
    
    print('determining whether a conflict took place or not')

    # select the entries which occured in this year
    temp_sel_year = conflict_gdf.loc[conflict_gdf.year == sim_year]   
    
    # merge the dataframes with polygons and conflict information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, extent_gdf)
    
    # determine the aggregated amount of fatalities in one region (e.g. water province)
    fatalities_per_watProv = data_merged['best'].groupby(data_merged['watprovID']).sum().to_frame().rename(columns={"best": 'total_fatalities'})
 
    # loop through all regions and check if exists in sub-set
    # if so, this means that there was conflict and thus assign value 1
    list_out = []
    for i in range(len(extent_gdf)):
        i_watProv = extent_gdf.iloc[i]['watprovID']
        if i_watProv in fatalities_per_watProv.index.values:
            list_out.append(1)
        else:
            list_out.append(0)
            
    if not len(extent_gdf) == len(list_out):
        raise AssertionError('the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_out)))

    return list_out