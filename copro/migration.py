#Change all conflict references to migration. ALso, delete code refering to conflict T-1 and conflict in neighbouring countries

from copro import utils, data
import geopandas as gpd
import pandas as pd
import numpy as np
import os, sys
import click


def migration_in_year_int(config, migration_gdf, extent_gdf, sim_year, out_dir): 
    """Creates a list for each timestep with integer information on migration in a polygon."

    Args: config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        # migration_gdf (geodataframe): geo-dataframe containing georeferenced information of migration.
        
extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
        sim_year (int): year for which data is extracted.
        out_dir (str): path to output folder. If 'None', no output is stored.

    Raises: AssertionError: raised if the length of output list does not match length of input geo-dataframe.
    Returns:
        list: list containing int per polygon depending on net migration.
   """
    
    if config.getboolean('general', 'verbose'): print('DEBUG: checking for migration in polygon at t')

    # select the entries which occured in this year
    temp_sel_year = migration_gdf.loc[migration_gdf.year == sim_year]  

    if len(temp_sel_year) == 0:
        click.echo('WARNING: no migration occured in sampled migration data set for year {}'.format(sim_year))
    
    # merge the dataframes with polygons and migration information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, extent_gdf)

    # DELETE determine the aggregated amount of fatalities in one region (e.g. water province)

    # DELETE fatalities_per_poly = data_merged['best'].groupby(data_merged['watprovID']).sum().to_frame().rename(columns={"best": 'total_fatalities'})

    out_dir = os.path.join(out_dir, 'files')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if sim_year == config.getint('settings', 'y_end'):
        # get the migration value for each polygon
        int_per_poly = 'net_migration'
        # change column name and dtype
        int_per_poly = int_per_poly.rename(columns={'int_migration'}).astype(int)
        # change index name to fit global_df
        int_per_poly.index = int_per_poly.index.rename('ID')
        # get list of all polygon IDs with their geometry information
        global_df = utils.global_ID_geom_info(extent_gdf)
        # merge the integer info with geometry
        # for all polygons without net migration, set a 0
        if config.getboolean('general', 'verbose'): print('DEBUG: storing integer migration map of year {} to file {}'.format(sim_year, os.path.join(out_dir, 'migration_in_{}.csv'.format(sim_year))))
        # data_stored = pd.merge(int_per_poly, global_df, on='ID', how='right').fillna(0)
        data_stored = pd.merge(int_per_poly, global_df, on='ID', how='right').dropna()
        data_stored.index = data_stored.index.rename('net_migration')
        data_stored = data_stored.drop('geometry', axis=1)
        data_stored = data_stored.astype(int)
        data_stored.to_csv(os.path.join(out_dir, 'migration_in_{}.csv'.format(sim_year)))
 
    # loop through all regions and check if exists in sub-set
    list_out = []
    for i in range(len(extent_gdf)):
        i_poly = extent_gdf.iloc[i]['GID_2']
        # DELETE if i_poly in fatalities_per_poly.index.values:
           # list_out.append(1)
        # else:
            # list_out.append(0)
            
    assert (len(extent_gdf) == len(list_out)), AssertionError('ERROR: the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_out)))

    return list_out

# DELETE STEP ON CONFLICT in previous year
# def conflict_in_previous_year(config, conflict_gdf, extent_gdf, sim_year, check_neighbors=False, neighboring_matrix=None):
    # Creates a list for each timestep with boolean information whether a conflict took place in a polygon at the previous timestep or not.
    # If the current time step is the first (t=0), then this year is skipped and the model continues at the next time step.

    # Args:
       # config (ConfigParser-object): object containing the parsed configuration-settings of the model.
       # conflict_gdf (geodataframe): geo-dataframe containing georeferenced information of conflict (tested with PRIO/UCDP data).
       # extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
       # sim_year (int): year for which data is extracted.
       # check_neighbors (bool): whether to check conflict events in neighboring polygons. Defaults to False.
       # neighboring_matrix (dataframe): lookup-dataframe indicating which polygons are mutual neighbors. Defaults to None.

   # Raises:
       # ValueError: raised if check_neighbors is True, but no matrix is provided.
       # AssertionError: raised if the length of output list does not match length of input geo-dataframe.

   # Returns:
       # list: list containing 0/1 per polygon depending on conflict occurence if checkinf for conflict at t-1, and containing log-transformed number of conflict events in neighboring polygons if specified.
   
    # if config.getboolean('general', 'verbose'): 
       # if check_neighbors: print('DEBUG: checking for conflicts in neighboring polygons at t-1')
       # else: print('DEBUG: checking for conflict event in polygon at t-1')

    # get conflicts at t-1
    # temp_sel_year = conflict_gdf.loc[conflict_gdf.year == sim_year-1]  

    # assert (len(temp_sel_year) != 0), AssertionError('ERROR: no conflicts were found in sampled conflict data set for year {}'.format(sim_year-1))
    
    # merge the dataframes with polygons and conflict information, creating a sub-set of polygons/regions
    # data_merged = gpd.sjoin(temp_sel_year, extent_gdf)
    
    # conflicts_per_poly = data_merged.id.groupby(data_merged['watprovID']).count().to_frame().rename(columns={"id": 'conflict_count'})

    # loop through all polygons and check if exists in sub-set
    #list_out = []
    # for i in range(len(extent_gdf)):

      #  i_poly = extent_gdf.watprovID.iloc[i]

      #  if i_poly in conflicts_per_poly.index.values:

          #  if check_neighbors:

            # determine log-scaled number of conflict events in neighboring polygons
            #  val = calc_conflicts_nb(i_poly, neighboring_matrix, conflicts_per_poly)
            # append resulting value
            # list_out.append(val)

           # else:

             #   list_out.append(1)

        # else:

        # if polygon not in list with conflict polygons, assign 0
        # list_out.append(0)
            
    # assert (len(extent_gdf) == len(list_out)), AssertionError('ERROR: the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_out)))

   #  return list_out

def read_projected_migration(extent_gdf, int_migration): # DELETE check_neighbors=False, neighboring_matrix=None)
    """Creates a list for each timestep with integer information on migration per polygon.
    Input migratation data (int_migration) must contain an index with IDs corresponding with the 'watprovID' values of extent_gdf. #waterprovID to be adapted
    Optionally, the algorithm can be extended to the neighboring polygons.

    Args:
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
        int_migration (dataframe): dataframe with integer values per polygon on net migration.
        # DELETE check_neighbors (bool, optional): whether or not to check for conflict in neighboring polygons. Defaults to False.
        # DELETE neighboring_matrix (dataframe, optional): look-up dataframe listing all neighboring polygons. Defaults to None.

    Returns:
        list: containing net migration values for each polygon. # DELETE If check_neighbors=True, then 1 if neighboring polygon contains conflict and 0 is not.
    """

        # assert that there are actually conflicts reported
    assert (len(int_migration) != 0), AssertionError('ERROR: no migration was found in sampled migration data set for year {}'.format(sim_year-1))

    # loop through all polygons and check if exists in sub-set
    list_out = []
    for i in range(len(extent_gdf)):

        i_poly = extent_gdf.GID_2.iloc[i] # change in GID_2

        if i_poly in int_migration.index.values:

            # DELETE if check_neighbors:

                # determine log-scaled number of conflict events in neighboring polygons
                # val = calc_conflicts_nb(i_poly, neighboring_matrix, bool_conflict)
                # append resulting value
                # list_out.append(val)

            # else:
            list_out.append(1)

        else:

            # if polygon not in list with conflict polygons, assign 0
            list_out.append(0)

    return list_out

# DELETE THIS PART ON CONFLICT IN NEIGHBOURING COUNTRIES
# def calc_conflicts_nb(i_poly, neighboring_matrix, conflicts_per_poly):
    # Determines whether in the neighbouring polygons of a polygon i_poly conflict took place.
    # If so, a value 1 is returned, otherwise 0.

    # Args:
       # i_poly (int): ID number of polygon under consideration.
       #neighboring_matrix (dataframe): look-up dataframe listing all neighboring polygons.
       # conflicts_per_poly (dataframe): dataframe with conflict informatoin per polygon.

    #Returns:
        # int: 1 is conflict took place in neighboring polygon, 0 if not.
    
    # find neighbors of this polygon
    # nb = data.find_neighbors(i_poly, neighboring_matrix)

    # initiate list
    # nb_count = []

    # loop through neighbors
    # for k in nb:

        # check if there was conflict at t-1
        # if k in conflicts_per_poly.index.values:

            # nb_count.append(1)

    # if more than one neighboring polygon has conflict, return 0
    # if np.sum(nb_count) > 0: 
       # val = 1
    # otherwise, return 0
   # else: 
      #  val = 0

    #return val

def get_poly_ID(extent_gdf): 
    """Extracts and returns a list with unique identifiers for each polygon used in the model. The identifiers are currently limited to 'watprovID'. # to be changed to GID_2

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
        list_ID.append(extent_gdf.iloc[i]['GID_2'])

    # in the end, the same number of polygons should be in geodataframe and list        
    assert (len(extent_gdf) == len(list_ID)), AssertionError('ERROR: the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_ID)))
        
    return list_ID

def get_poly_geometry(extent_gdf, config): 
    """Extracts geometry information for each polygon from geodataframe and saves to list. The geometry column in geodataframe must be named 'geometry'."""

    """Args:
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
    assert (len(extent_gdf) == len(list_geometry)), AssertionError('ERROR: the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(extent_gdf), len(list_geometry)))
        
    return list_geometry

def split_migration_geom_data(X):
    # Separates the unique identifier, geometry information, and data from the variable-containing X-array.

    """Args:
        X (array): variable-containing X-array.

    Returns:
        arrays: seperate arrays with ID, geometry, and actual data 
    
    # first column corresponds to ID, second to geometry
    # all remaining columns are actual data
    X_ID = X[:, 0]
    X_geom = X[:, 1]
    X_data = X[: , 2:]

    return X_ID, X_geom, X_data
"""
def get_pred_migration_geometry(X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1):
    # Stacks together the arrays with unique identifier, geometry, test data, and predicted data into a dataframe. 
    #Contains therefore only the data points used in the test-sample, not in the training-sample. 
    # Additionally computes whether a correct prediction was made.

    """Args:
        X_test_ID (list): list containing the unique identifier per data point.
        X_test_geom (list): list containing the geometry per data point.
        y_test (list): list containing test-data.
        y_pred (list): list containing predictions.

    Returns:
        dataframe: dataframe with each input list as column plus computed 'correct_pred'.
"""
    # stack separate columns horizontally
    arr = np.column_stack((X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1))

    # convert array to dataframe
    df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_test', 'y_pred', 'y_prob_0', 'y_prob_1'])

    # compute whether a prediction is correct
    # if so, assign 1; otherwise, assign 0
    df['correct_pred'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

    return df
