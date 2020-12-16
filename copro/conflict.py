from copro import data
import geopandas as gpd
import pandas as pd
import numpy as np
import os, sys
import math

def conflict_in_year_bool(config, conflict_gdf, extent_gdf, sim_year): 
    """Creates a list for each timestep with boolean information whether a conflict took place in a polygon or not.

    Args:
        conflict_gdf (geodataframe): geo-dataframe containing georeferenced information of conflict (tested with PRIO/UCDP data).
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
        sim_year (int): year for which data is extracted.

    Raises:
        AssertionError: raised if the length of output list does not match length of input geo-dataframe.

    Returns:
        list: list containing 0/1 per polygon depending on conflict occurence.
    """    

    # select the entries which occured in this year
    temp_sel_year = conflict_gdf.loc[conflict_gdf.year == sim_year]   
    
    # merge the dataframes with polygons and conflict information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, extent_gdf)
    
    # determine the aggregated amount of fatalities in one region (e.g. water province)
    try:
        fatalities_per_poly = data_merged['best'].groupby(data_merged['watprovID']).sum().to_frame().rename(columns={"best": 'total_fatalities'})
    except:
        fatalities_per_poly = data_merged['best'].groupby(data_merged['name']).sum().to_frame().rename(columns={"best": 'total_fatalities'})
 
    # loop through all regions and check if exists in sub-set
    # if so, this means that there was conflict and thus assign value 1
    list_out = []
    for i in range(len(extent_gdf)):
        try:
            i_poly = extent_gdf.iloc[i]['watprovID']
        except:
            i_poly = extent_gdf.iloc[i]['name']
        if i_poly in fatalities_per_poly.index.values:
            list_out.append(1)
        else:
            list_out.append(0)
            
    if not len(extent_gdf) == len(list_out):
        raise AssertionError('the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_out)))

    return list_out

def conflict_in_previous_year(config, conflict_gdf, extent_gdf, sim_year, check_neighbors=False, neighboring_matrix=None):
    """Creates a list for each timestep with boolean information whether a conflict took place in a polygon at the previous timestep or not.
    If the current time step is the first (t=0), then conflict data of this year is used instead due to the lack of earlier data.

    Args:
        conflict_gdf (geodataframe): geo-dataframe containing georeferenced information of conflict (tested with PRIO/UCDP data).
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
        sim_year (int): year for which data is extracted.
        check_neighbors (bool): whether to check conflict events in neighboring polygons. Defaults to False.
        neighboring_matrix (dataframe): lookup-dataframe indicating which polygons are mutual neighbors. Defaults to None.

    Raises:
        ValueError: raised if check_neighbors is True, but no matrix is provided.
        AssertionError: raised if the length of output list does not match length of input geo-dataframe.

    Returns:
        list: list containing 0/1 per polygon depending on conflict occurence if checkinf for conflict at t-1, and containing log-transformed number of conflict events in neighboring polygons if specified.
    """    

    # get conflicts at t-1
    temp_sel_year = conflict_gdf.loc[conflict_gdf.year == sim_year-1]  
    
    # merge the dataframes with polygons and conflict information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, extent_gdf)

    # determine log-transformed count of unique conflicts per water province
    # the id column refers to the conflict id, not the water province id!
    if config.getboolean('general', 'verbose'): 
        if check_neighbors: print('DEBUG: computing log-transformed count of conflicts in neighboring polygons at t-1')
        else: print('DEBUG: checking for conflict event in polygon at t-1')

    conflicts_per_poly = data_merged.id.groupby(data_merged['watprovID']).count().to_frame()

    # loop through all polygons and check if exists in sub-set
    list_out = []
    for i in range(len(extent_gdf)):

        i_poly = extent_gdf.watprovID.iloc[i]

        if i_poly in conflicts_per_poly.index.values:

            if check_neighbors:

                # if neighboring_matrix == None:
                #     raise ValueError('ERROR: a valid lookup matrix for neighboring polygons must be provided if model is ought to check for conflicts in neighboring polygons!')

                # determine log-scaled number of conflict events in neighboring polygons
                val = calc_conflicts_nb(config, i_poly, neighboring_matrix, conflicts_per_poly)
                # append resulting value
                list_out.append(val)

            else:

                list_out.append(1)

        else:

            # if polygon not in list with conflict polygons, assign 0
            list_out.append(0)
            
    if not len(extent_gdf) == len(list_out):
        raise AssertionError('the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_out)))

    return list_out

def calc_conflicts_nb(config, i_poly, neighboring_matrix, conflicts_per_poly):
    """[summary]

    Args:
        config ([type]): [description]
        i_poly ([type]): [description]
        neighboring_matrix ([type]): [description]
        conflicts_per_poly ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # find neighbors of this polygon
    nb = data.find_neighbors(i_poly, neighboring_matrix)

    # initialize count for total numbers of conflicts in neighbors
    tot_nr_confl = 0.0

    # loop through neighbors
    for k in nb:

        # check if there was conflict at t-1
        if k in conflicts_per_poly.index.values:

            # determine number of conflicts per neigbors at t-1
            val = conflicts_per_poly.id.loc[conflicts_per_poly.index == k].values[0]

            # add to sum
            tot_nr_confl += val

    # log-transform value
    if config.getboolean('general', 'verbose'): print('DEBUG: total number of conflicts at t-1 for watprovID {} is {}'.format(i_poly, tot_nr_confl))
    tot_nr_confl = np.log(tot_nr_confl)

    # if log-transformed value is -inf, mask with zero
    if tot_nr_confl == -math.inf:

        print('WARNING: no -inf allowed - setting value for watprovID {} to 0'.format(i_poly))
        tot_nr_confl = 0.0

    return tot_nr_confl

def get_poly_ID(extent_gdf): 
    """Extracts and returns a list with unique identifiers for each polygon used in the model. The identifiers are currently limited to 'name' or 'watprovID'.

    Args:
        extent_gdf (geo-dataframe): geo-dataframe containing one or more polygons.

    Raises:
        AssertionError: error raised if length of output list does not match length of input geo-dataframe.

    Returns:
        list: list containing a unique identifier extracted from geo-dataframe for each polygon used in the model.
    """  

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

def get_poly_geometry(extent_gdf, config): 
    """Extracts geometry information for each polygon from geodataframe and saves to list. The geometry column in geodataframe must be named 'geometry'.

    Args:
        extent_gdf (geo-dataframe): geo-dataframe containing one or more polygons with geometry information.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Raises:
        AssertionError: error raised if length of output list does not match length of input geo-dataframe.

    Returns:
        list: list containing the geometry information extracted from geo-dataframe for each polygon used in the model.
    """    
    
    if config.getboolean('general', 'verbose'): print('DEBUG: getting the geometry of all geographical units')

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
    """Separates the unique identifier and geometry information from the variable-containing X-array.

    Args:
        X (array): variable-containing X-array.

    Returns:
        arrays: seperate arrays with ID, geometry, and  actual data 
    """    

    X_ID = X[:, 0]
    X_geom = X[:, 1]
    X_data = X[: , 2:]

    return X_ID, X_geom, X_data

def get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred):
    """Stacks together the arrays with unique identifier, geometry, test data, and predicted data into a dataframe. 
    Contains therefore only the data points used in the test-sample, not in the training-sample. 
    Additionally computes whether a correct prediction was made in column 'correct_pred'.

    Args:
        X_test_ID (list): list containing the unique identifier per data point.
        X_test_geom (list): list containing the geometry per data point.
        y_test (list): list containing test-data.
        y_pred (list): list containing predictions.

    Returns:
        dataframe: dataframe with each input list as column plus computed 'correct_pred'.
    """   

    arr = np.column_stack((X_test_ID, X_test_geom, y_test, y_pred))

    df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_test', 'y_pred'])

    df['correct_pred'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

    return df

