from copro import utils, data
import geopandas as gpd
import pandas as pd
import numpy as np
import os, sys
import click


def migration_in_year_int(config, migration_gdf, gdf, sim_year, out_dir): 
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
    temp_sel_year.to_csv(os.path.join(out_dir, 'temp_sel_year_in_{}.csv'.format(sim_year)))

    if len(temp_sel_year) == 0:
        click.echo('WARNING: no migration occured in sampled migration data set for year {}'.format(sim_year))
  
    # merge the dataframes with polygons and migration information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, migration_gdf)
    data_merged.to_csv(os.path.join(out_dir, 'data_merged_in_{}.csv'.format(sim_year)))

    out_dir = os.path.join(out_dir, 'files')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if sim_year == config.getint('settings', 'y_end'):
    
        # get the migration value for each polygon
        int_per_poly = temp_sel_year.copy()  
        # DELETE change column name - seems unnecessary  
        # DELETE int_per_poly = int_per_poly.rename(columns={"net_migration": "int_migration"}) # check if this renaming is still necessary, doesnt seem to be the case
        int_per_poly = int_per_poly.rename(columns={int_per_poly.columns[0]: 'GID_2'}) # check if what happens here is correct
        # change index name to fit global_df --> 
        int_per_poly = int_per_poly.reset_index().rename(columns={'GID_2': 'ID'})
        # int_per_poly.index = int_per_poly.index.rename('ID')
      
        # get list of all polygon IDs with their geometry information
        global_df = utils.global_ID_geom_info(gdf)
        # for all polygons without net migration, set a 0
        if config.getboolean('general', 'verbose'): print('DEBUG: storing integer migration map of year {} to file {}'.format(sim_year, os.path.join(out_dir, 'migration_in_{}.csv'.format(sim_year))))
        int_per_poly['ID'] = int_per_poly['ID'].astype(object)
        data_stored = pd.merge(int_per_poly, global_df, on='ID', how='right').fillna(0)
        # data_stored.index = data_stored.index.rename('GID_2')
        data_stored = data_stored.drop(columns=['geometry_x', 'geometry_y'])
        data_stored.to_csv(os.path.join(out_dir, 'migration_in_{}.csv'.format(sim_year)))
    column_name = 'GID_2'
    has_column = column_name in gdf.columns
       


    # loop through all regions and check if exists in sub-set
    list_out = []
    for i in range(len(gdf)):
        i_poly = gdf.iloc[i]['GID_2']
        if i_poly in gdf['GID_2'].values:
            migration_value = gdf.loc[(gdf['GID_2'] == i_poly) & (gdf['year'] == sim_year), 'net_migration'].values[0]
            list_out.append(migration_value)
        else:
    # Handle the case when the value is not found
            migration_value = None  # or assign a default value
            print(f"WARNING: The value {i_poly} is not present in the 'GID_2' column of gdf.")

    # for i in range(len(gdf)):
       # i_poly = gdf.iloc[i]['GID_2']
        #migration_value = data_merged.loc[gdf['GID_2'] == i_poly, sim_year].values[0] # to be adapted
        #list_out.append(migration_value)
        #print('migration assigned')
            
    assert (len(gdf) == len(list_out)), AssertionError('ERROR: the dataframe with polygons has a lenght {0} while the lenght of the resulting list is {1}'.format(len(gdf), len(list_out)))

    return list_out

def read_projected_migration(extent_gdf, net_migration): # DELETE check_neighbors=False, neighboring_matrix=None)
    """Creates a list for each timestep with integer information on migration per polygon.
    Input migratation data (net_migration) must contain an index with IDs corresponding with the 'GID_2' values of the gdf. 
    Optionally, the algorithm can be extended to the neighboring polygons.

    Args:
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
        net_migration (dataframe): dataframe with integer values per polygon on net migration.
        # DELETE check_neighbors (bool, optional): whether or not to check for conflict in neighboring polygons. Defaults to False.
        # DELETE neighboring_matrix (dataframe, optional): look-up dataframe listing all neighboring polygons. Defaults to None.

    Returns:
        list: containing net migration values for each polygon. # DELETE If check_neighbors=True, then 1 if neighboring polygon contains conflict and 0 is not.
    """

        # assert that there are actually conflicts reported
    assert (len(net_migration) != 0), AssertionError('ERROR: no migration was found in sampled migration data set for year {}'.format(sim_year-1))

    # loop through all polygons and check if exists in sub-set
    list_out = []
    for i in range(len(extent_gdf)):

        i_poly = extent_gdf.GID_2.iloc[i] 

        if i_poly in net_migration.index.values:

            list_out.append(1) # should this be changed to just give the integer value? 

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
    """Extracts and returns a list with unique identifiers for each polygon used in the model. The identifier is in this version limited to 'GID_2', can be adapted to the identifier one has.

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
 """   
    #first column corresponds to ID, second to geometry
    #all remaining columns are actual data
    X_ID = X[:, 0]
    X_geom = X[:, 1]
    X_data = X[: , 2:]

    return X_ID, X_geom, X_data

def get_pred_migration_geometry_classifier(X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1):
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

def get_pred_migration_geometry_regression(X_test_ID, X_test_geom, y_test, y_pred):
    # Stacks together the arrays with unique identifier, geometry, test data, and predicted data into a dataframe.
    # Contains only the data points used in the test-sample, not in the training-sample.
    # Additionally computes whether a correct prediction was made.

    """
    Args:
        X_test_ID (list): list containing the unique identifier per data point.
        X_test_geom (list): list containing the geometry per data point.
        y_test (list): list containing test-data.
        y_pred (list): list containing predictions.

    Returns:
        dataframe: dataframe with each input list as a column plus computed 'correct_pred'.
    """
    # stack separate columns horizontally
    arr = np.column_stack((X_test_ID, X_test_geom, y_test, y_pred))

    # convert array to dataframe
    df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_test', 'y_pred'])

    # compute whether a prediction is correct
    # since this is regression, there is no exact match, so no 'correct_pred' column is computed

    return df

