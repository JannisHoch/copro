import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

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
    try:
        fatalities_per_watProv = data_merged['best'].groupby(data_merged['watprovID']).sum().to_frame().rename(columns={"best": 'total_fatalities'})
    except:
        fatalities_per_watProv = data_merged['best'].groupby(data_merged['name']).sum().to_frame().rename(columns={"best": 'total_fatalities'})
 
    # loop through all regions and check if exists in sub-set
    # if so, this means that there was conflict and thus assign value 1
    list_out = []
    for i in range(len(extent_gdf)):
        try:
            i_watProv = extent_gdf.iloc[i]['watprovID']
        except:
            i_watProv = extent_gdf.iloc[i]['name']
        if i_watProv in fatalities_per_watProv.index.values:
            list_out.append(1)
        else:
            list_out.append(0)
            
    if not len(extent_gdf) == len(list_out):
        raise AssertionError('the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_out)))

    return list_out

def get_poly_ID(extent_gdf): 
    """Extracts geometry information for each polygon from geodataframe and saves in list.

    Args:
        extent_gdf ([type]): [description]

    Raises:
        AssertionError: [description]

    Returns:
        [type]: [description]
    """    
    
    print('listing the geometry of all geographical units')

    # initiatie empty list
    list_ID = []

    # loop through all polygons
    for i in range(len(extent_gdf)):
        # append geometry of each polygon to list
        try:
            list_ID.append(extent_gdf.iloc[i]['name'])
        except:
            list_ID.append(extent_gdf.iloc[i]['watprovID'])

    # in the end, the same number of polygons should be in geodataframe and list        
    if not len(extent_gdf) == len(list_ID):
        raise AssertionError('the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_ID)))
        
    return list_ID

def get_poly_geometry(extent_gdf): 
    """Extracts geometry information for each polygon from geodataframe and saves in list.

    Args:
        extent_gdf ([type]): [description]

    Raises:
        AssertionError: [description]

    Returns:
        [type]: [description]
    """    
    
    print('listing the geometry of all geographical units')

    # initiatie empty list
    list_geometry = []

    # loop through all polygons
    for i in range(len(extent_gdf)):
        # append geometry of each polygon to list
        list_geometry.append(extent_gdf.iloc[i]['geometry'])

    # in the end, the same number of polygons should be in geodataframe and list        
    if not len(extent_gdf) == len(list_geometry):
        raise AssertionError('the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_geometry)))
        
    return list_geometry

def split_conflict_geom_data(X):
    """[summary]

    Args:
        X ([type]): [description]

    Returns:
        [type]: [description]
    """    

    X_ID = X[:, 0]
    X_geom = X[:, 1]
    X_data = X[: , 2:]

    return X_ID, X_geom, X_data

def get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred):
    """[summary]

    Args:
        X_test_geom ([type]): [description]
        y_test ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """   

    arr = np.column_stack((X_test_ID, X_test_geom, y_test, y_pred))

    df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_test', 'y_pred'])

    df['conflict_hit'] = np.where((df['y_test'] == 1) & (df['y_pred'] ==1), 1, np.nan)

    df['overall_hit'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

    gdf = gpd.GeoDataFrame(df, geometry=df.geometry)

    return df, gdf