from copro import utils, data
import geopandas as gpd
import pandas as pd
import numpy as np
import os, sys
import click
import shapely
from shapely.wkt import loads
from scipy.stats.mstats import winsorize


def migration_in_year_int(root_dir, config, migration_gdf, extent_gdf, sim_year, out_dir): 
    """Creates a list for each timestep with integer information on migration in a polygon, or if indicated in the cfg file a weightened list based on the total population per polygon."

    Args: config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): absolute path to location of configurations-file
        migration_gdf (geodataframe): geo-dataframe containing georeferenced information of migration.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
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

    out_dir = os.path.join(out_dir, 'files')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if config.getboolean('general', 'verbose'): print('DEBUG: check if migration should be set to % based on population')
    # check if migration should be in %:
    if config.getboolean('migration', 'migration_percentage'): 
        total_population_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('migration', 'total_population'))
        total_population = pd.read_csv(total_population_fo)
        # select the total population per polygon this year
        population_total_sel_year = total_population.loc[total_population['year'] == sim_year]

        combined_migration_data = temp_sel_year.merge(population_total_sel_year, on='GID_2', how='left')
        combined_migration_data['migration_perc'] = combined_migration_data['net_migration'] / combined_migration_data['total_population']

        # drop 'old' net_migration column
        combined_migration_data.drop(columns='net_migration', inplace=True)

        # Rename the column 'weighted_migration' to 'net_migration'
        combined_migration_data.rename(columns={'migration_perc': 'net_migration'}, inplace=True)

        if config.getboolean('general', 'verbose'): print('DEBUG: storing migration % csv of year {} to file {}'.format(sim_year, os.path.join(out_dir, 'net_migration%_in_{}.csv'.format(sim_year))))
        combined_migration_data_exgeo = combined_migration_data.drop(columns='geometry') 
        combined_migration_data_exgeo.to_csv(os.path.join(out_dir, 'migration_in_{}.csv'.format(sim_year)))

        temp_sel_year = combined_migration_data
    else:
        pass

    if sim_year == config.getint('settings', 'y_end'):
    
        # get the migration value for each polygon
        int_per_poly = temp_sel_year.copy()  

        if config.getboolean('general', 'verbose'): print('DEBUG: storing integer migration csv of year {} to file {}'.format(sim_year, os.path.join(out_dir, 'migration_in_{}.csv'.format(sim_year))))

        int_per_poly.to_csv(os.path.join(out_dir, 'migration_in_{}.csv'.format(sim_year))) 
            
    # loop through all regions and check if exists in sub-set
    list_out = []

    # select the polygons that must be selected
    polygon_names = extent_gdf['GID_2'].unique().tolist()

    selected_migration_data = temp_sel_year[temp_sel_year['GID_2'].isin(polygon_names)]

    selected_data = selected_migration_data.copy()
   
    list_out.extend(zip(selected_data['net_migration'], selected_data['GID_2'].values.tolist()))

    return list_out

def read_projected_migration(extent_gdf, net_migration):
    """Creates a list for each timestep with integer information on migration per polygon.
    Input migratation data (net_migration) must contain an index with IDs corresponding with the 'GID_2' values of the gdf. 
    Optionally, the algorithm can be extended to the neighboring polygons.

    Args:
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons with geometry information for which values are extracted.
        net_migration (dataframe): dataframe with integer values per polygon on net migration.

    Returns:
        list: containing net migration values for each polygon. # DELETE If check_neighbors=True, then 1 if neighboring polygon contains conflict and 0 is not.
    """

    # loop through all polygons and check if exists in sub-set
    list_out = []
    for i in range(len(extent_gdf)):

        i_poly = extent_gdf.GID_2.iloc[i] 

        if i_poly in net_migration.index.values:

            list_out.append(1)  

        else:

            # if polygon not in list with conflict polygons, assign 0
            list_out.append(0)

    return list_out


def get_poly_ID(extent_gdf): 
    """Extracts and returns a list with unique identifiers for each polygon used in the model. The identifier is in this version limited to 'GID_2', can be adapted to the identifier one has.

    Args:
        extent_gdf (geo-dataframe): geo-dataframe containing one or more polygons.

    Raises:
        AssertionError: error raised if length of output list does not match length of input geo-dataframe.

    Returns:
        list: list containing a unique identifier extracted from geo-dataframe for each polygon used in the model. 
"""
     # initiate an empty set to store unique identifiers
    unique_ids = set()

    # loop through all polygons
    for i in range(len(extent_gdf)):
        # get the identifier for the current polygon
        identifier = extent_gdf.iloc[i]['GID_2']

        # check if the identifier has already been added to the set
        if identifier not in unique_ids:
            # if not, append it to the list_ID and add it to the set
            unique_ids.add(identifier)

    # convert the set back to a list
    list_ID = list(unique_ids)
        
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

    # initiate an empty set to store unique geometry representations
    unique_geometries = set()

    # loop through all polygons
    for i in range(len(extent_gdf)):
        # get the geometry of the current polygon
        geometry = extent_gdf.iloc[i]['geometry']

        # add the geometry's string representation to the set (it will only be added if it's unique)
        unique_geometries.add(str(geometry))

    # convert the set back to a list of geometries
    list_geometry = [shapely.wkt.loads(geometry_str) for geometry_str in unique_geometries]
    
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

def get_pred_migration_geometry_classifier(X_test_ID, y_test, y_pred, y_prob_0, y_prob_1): # deleted X_test_geom, 
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
    arr = np.column_stack((X_test_ID, y_test, y_pred, y_prob_0, y_prob_1)) # deleted X_test_geom

    # convert array to dataframe
    df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_test', 'y_pred', 'y_prob_0', 'y_prob_1'])

    # compute whether a prediction is correct
    # if so, assign 1; otherwise, assign 0
    df['correct_pred'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

    return df

def get_pred_migration_geometry_regression(X_test_ID, y_test, y_pred): # deleted X_test_geom, 
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
    arr = np.column_stack((X_test_ID, y_test, y_pred)) # X_test_geom

    # convert array to dataframe
    df = pd.DataFrame(arr, columns=['ID',  'y_test', 'y_pred']) # delete 'geometry'

    # compute whether a prediction is correct
    # since this is regression, there is no exact match, so no 'correct_pred' column is computed

    return df

def weight_migration(config, root_dir, migration_gdf):
        """ Args:
    config (ConfigParser-object): object containing the parsed configuration-settings of the model. 
    root_dir (str): absolute path to location of configurations-file
    migration_gdf (GeoDataFrame): GeoDataFrame containing migration data

    Returns:
    A variable with normalised_weights to weight the y_train data if indicated in the cfg-file
    """
        # Define the year range from the configuration
        y_start = config.getint('settings', 'y_start')
        y_end = config.getint('settings', 'y_end')
        years_range = list(range(y_start, y_end +1))

        total_population_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('migration', 'total_population'))
        total_population = pd.read_csv(total_population_fo)
        selected_population = total_population[(total_population['year'] >= y_start) & (total_population['year'] <= y_end +1)]
    
        merged_dfs = []

        # Get unique GID_2 values from migration_gdf
        unique_gid_2_values = migration_gdf['GID_2'].unique()

        for gid_2 in unique_gid_2_values:
            for year in years_range:
                # Create a DataFrame with all combinations of GID_2 and year
                combination_df = pd.DataFrame({'GID_2': [gid_2], 'year': [year]})

                # Merge combination_df with selected_total_population based on GID_2 and year
                merged_df = combination_df.merge(selected_population, on=['GID_2', 'year'], how='left')

                # Merge merged_df with migration_gdf based on GID_2 and year
                merged_df = merged_df.merge(migration_gdf, on=['GID_2', 'year'], how='left')

                # Append the merged DataFrame to the list
                merged_dfs.append(merged_df)

        # Concatenate all merged DataFrames to get the final result
        final_merged_df = pd.concat(merged_dfs, ignore_index=True)

        # calculate weights based on population
        weights = final_merged_df['total_population'].values

        # Winsorization threshold
        winsor_threshold = 0.5 # to discuss what a good value for this threshold we could take

        # Winsorize the weights
        winsorised_weights = winsorize(weights, limits=(0, winsor_threshold)) # to discuss, is this the best way to robustly weight the migration data? 
        # Create a DataFrame to store GID_2, year, and their corresponding weights
        gid2_weights = pd.DataFrame({'GID_2': final_merged_df['GID_2'],'year': final_merged_df['year'],'weight': winsorised_weights})

        return gid2_weights

