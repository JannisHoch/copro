from copro import utils, data
import geopandas as gpd
import pandas as pd
import numpy as np
import os, sys
import click
import shapely
from shapely.wkt import loads
from scipy.stats.mstats import winsorize


def migration_in_year_int(root_dir, config, migration_gdf, sim_year, out_dir): 
    """Creates a list for each timestep with integer information on migration in a polygon, or if indicated in the cfg file a weightened list based on the total population per polygon."

    Args: config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): absolute path to location of configurations-file
        migration_gdf (geodataframe): geo-dataframe containing georeferenced information of migration.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
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
        total_population_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('migration', 'population_total'))
        total_population = pd.read_csv(total_population_fo)
        # select the total population per polygon this year
        population_total_sel_year = total_population.loc[total_population['year'] == sim_year]
        # merge the two dataframes
        combined_migration_data = temp_sel_year.merge(population_total_sel_year, on='GID_2', how='left')
        #calculate the net migration percentage based on the total population per polygon
        combined_migration_data['migration_perc'] = combined_migration_data['net_migration'] / combined_migration_data['population_total']
        # drop 'old' net_migration column
        combined_migration_data.drop(columns='net_migration', inplace=True)
        # Rename the column 'weighted_migration' to 'net_migration'
        combined_migration_data.rename(columns={'migration_perc': 'net_migration'}, inplace=True)

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
    polygon_names = migration_gdf['GID_2'].unique().tolist()

    selected_migration_data = temp_sel_year[temp_sel_year['GID_2'].isin(polygon_names)]

    selected_data = selected_migration_data.copy()
   
    list_out.extend(zip(selected_data['net_migration'], selected_data['GID_2'].values.tolist()))

    return list_out

def migration_multiple_years(root_dir, config, migration_gdf, sim_year, out_dir): 
    """Creates a list for each timestep with integer information on migration in a polygon, or if indicated in the cfg file a weightened list based on the total population per polygon."

    Args: config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): absolute path to location of configurations-file
        migration_gdf (geodataframe): geo-dataframe containing georeferenced information of migration.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        sim_year (int): year for which data is extracted.
        out_dir (str): path to output folder. If 'None', no output is stored.

    Raises: AssertionError: raised if the length of output list does not match length of input geo-dataframe.
    Returns:
        list: list containing int per polygon depending on net migration.
   """
        
    # get years_to_average depending on the config settings
    if config.getboolean('general', 'three_year_migration_average'):
        years_to_average = [sim_year, sim_year + 1, sim_year + 2]                       
    
    elif config.getboolean('general', 'five_year_migration_average'):
        years_to_average = [sim_year, sim_year + 1, sim_year + 2, sim_year + 3, sim_year + 4]

    temp_sel_three_years = migration_gdf[migration_gdf['year'].isin(years_to_average)]

    if len(temp_sel_three_years) == 0:
        click.echo('WARNING: no migration occurred in sampled migration data set for years {}'.format(years_to_average))
  
    out_dir = os.path.join(out_dir, 'files')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if config.getboolean('general', 'verbose'):
        print('DEBUG: check if migration should be set to % based on population')

    if config.getboolean('migration', 'migration_percentage'): 
        total_population_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('migration', 'population_total'))
        total_population = pd.read_csv(total_population_fo)

        population_total_sel_years = total_population[total_population['year'].isin(years_to_average)]
        combined_migration_data = temp_sel_three_years.merge(population_total_sel_years, on=['GID_2', 'year'], how='left')

        combined_migration_data['migration_perc'] = combined_migration_data['net_migration'] / combined_migration_data['population_total']
        combined_migration_data.drop(columns=['net_migration', 'population_total'], inplace=True)
        combined_migration_data.rename(columns={'migration_perc': 'net_migration'}, inplace=True)

        temp_sel_three_years = combined_migration_data

    if sim_year == config.getint('settings', 'y_end'): # this isnot correct at the moment
        int_per_poly = temp_sel_three_years.copy()

        if config.getboolean('general', 'verbose'):
            print('DEBUG: storing integer migration csv for years {} to file {}'.format(years_to_average, os.path.join(out_dir, 'migration_in_{}_to_{}.csv'.format(years_to_average[0], years_to_average[-1]))))

        int_per_poly.to_csv(os.path.join(out_dir, 'migration_in_{}_to_{}.csv'.format(years_to_average[0], years_to_average[-1])))

    sum_per_poly = temp_sel_three_years.groupby('GID_2')['net_migration'].sum().reset_index()

    list_out = []
    for _, row in sum_per_poly.iterrows():
        list_out.append((row['net_migration'], row['GID_2']))

    return list_out

def get_poly_ID(migration_gdf): 
    """Extracts and returns a list with unique identifiers for each polygon used in the model. The identifier is in this version limited to 'GID_2', can be adapted to the identifier one has.

    Args:
        migration_gdf (geo-dataframe): geo-dataframe containing migration, polygon-geometry and polygon-ID information

    Raises:
        AssertionError: error raised if length of output list does not match length of input geo-dataframe.

    Returns:
        list: list containing a unique identifier extracted from geo-dataframe for each polygon used in the model. 
"""
     # initiate an empty set to store unique identifiers
    unique_ids = set()

    # loop through all polygons
    for i in range(len(migration_gdf)):
        # get the identifier for the current polygon
        identifier = migration_gdf.iloc[i]['GID_2']

        # check if the identifier has already been added to the set
        if identifier not in unique_ids:
            # if not, append it to the list_ID and add it to the set
            unique_ids.add(identifier)

    # convert the set back to a list
    list_ID = list(unique_ids)
        
    return list_ID

def get_poly_geometry(migration_gdf, config): 
    """Extracts geometry information for each polygon from geodataframe and saves to list. The geometry column in geodataframe must be named 'geometry'.

    Args:
        migration_gdf (geo-dataframe): geo-dataframe containing migration, polygon-geometry and polygon-ID information
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
    for i in range(len(migration_gdf)):
        # get the geometry of the current polygon
        geometry = migration_gdf.iloc[i]['geometry']

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

        total_population_fo = os.path.join(root_dir, config.get('PROJ_data', 'population_total'))
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
        weights = final_merged_df['population_total'].values

        # Winsorization threshold
        winsor_threshold = 0.5 # to discuss what a good value for this threshold we could take

        # Winsorize the weights
        winsorised_weights = winsorize(weights, limits=(0, winsor_threshold)) # to discuss, is this the best way to robustly weight the migration data? 
        # Create a DataFrame to store GID_2, year, and their corresponding weights
        gid2_weights = pd.DataFrame({'GID_2': final_merged_df['GID_2'],'year': final_merged_df['year'],'population_total': final_merged_df['population_total'], 'weight': winsorised_weights})

        return gid2_weights

def make_projections_population(config, config_REF, root_dir, proj_year, out_dir_PROJ, mdl):
    """ Args:
    config (ConfigParser-object): object containing the parsed configuration-settings of the model. 
    root_dir (str): absolute path to location of configurations-file.
    proj_year (int): year for which projection is made.
    out_dir_PROJ: (str) path to output folder for projection files.
    mdl: the specified model instance.

    Returns:
    A (geo)dataframe with the new population per polygon, based on population t-1, population growth t-1 and net migration t-1
    """   

    projection_year_min1 = int(config.get('settings', 'y_end'))
    # for the first year, we need to calculate the new population per polygon and store this in the output folder
    if proj_year == int(config.get('settings', 'y_end')) + 1:
        if config.getboolean('general', 'verbose'):
            print('DEBUG: calculating and storing total population per polygon for the first projection year')
            
            # get the total population for each polygon
            tot_population_path = os.path.join(root_dir, config_REF.get('POP_data', 'population_total'))
            tot_population = pd.read_csv(tot_population_path)
            population_total_last_year = tot_population.loc[tot_population['year'] == projection_year_min1]

            # get population growth for each polygon
            population_growth_fo = os.path.join(root_dir, config_REF.get('POP_data', 'population_growth'))
            population_growth = pd.read_csv(population_growth_fo)
            population_growth_last_year = population_growth.loc[population_growth['year'] == projection_year_min1]

            # get the net migration for each polygon
            migration_gdf = utils.get_geodataframe(config, root_dir)
            migration_last_year = migration_gdf.loc[migration_gdf['year'] == projection_year_min1]
            
            # Merge population_total_last_year, population_growth_last_year, and migration_last_year DataFrames based on 'GID_2'
            merged_df = pd.merge(population_total_last_year[['GID_2', 'population_total']], migration_last_year[['GID_2', 'net_migration']], on='GID_2', how='inner')
            merged_df = pd.merge(merged_df, population_growth_last_year[['GID_2', 'population_growth']], on='GID_2', how='inner', suffixes=('_pop', '_growth'))
            merged_df.to_csv(os.path.join(out_dir_PROJ, 'all_indicators_for_{}_exgeo.csv'.format(proj_year)))
            
            # calculate new population per polygon based on the population in the former year, the population growth and the net migration
            merged_df['new_population_per_polygon'] = merged_df['population_total'] + (merged_df['population_total'] * merged_df['population_growth']) + (merged_df['population_total'] * (merged_df['net_migration'] / merged_df['population_total']))
            merged_df.rename(columns={'population_total': 'population_t-1'}, inplace=True)
            merged_df.to_csv(os.path.join(out_dir_PROJ, 'population_for_{}_exgeo.csv'.format(proj_year)))
    
    # for the following years, we can use the calculated new population and % net migration per polygon as the input to calculate the new total population per polygon
    else:
        if config.getboolean('general', 'verbose'):
            print('DEBUG: calculating and storing total population per polygon for {}'.format(proj_year))
            proj_year_min1 = proj_year - 1 
            
            # get the total population for each polygon
            tot_population_path = os.path.join(out_dir_PROJ, 'population_for_{}_exgeo.csv'.format(proj_year_min1))
            population_total_last_year = pd.read_csv(tot_population_path)
            
            # get population growth for each polygon
            population_growth_fo = os.path.join(root_dir, config_REF.get('POP_data', 'population_growth'))
            population_growth = pd.read_csv(population_growth_fo)
            population_growth_last_year = population_growth.loc[population_growth['year'] == proj_year_min1]
    
            migration_last_year_fo = os.path.join(out_dir_PROJ, 'mdls', str(mdl).rsplit('.')[0], 'projection_for_{}_exgeo.csv'.format(proj_year_min1))
            migration_last_year = pd.read_csv(migration_last_year_fo)  
            migration_last_year.rename(columns={'ID': 'GID_2'}, inplace=True)

            # Merge population_total_last_year, population_growth_last_year, and migration_last_year DataFrames based on 'GID_2'
            merged_df = pd.merge(population_total_last_year[['GID_2', 'new_population_per_polygon']], migration_last_year[['GID_2', 'y_pred']], on='GID_2', how='inner')
            merged_df = pd.merge(merged_df, population_growth_last_year[['GID_2', 'population_growth']], on='GID_2', how='inner', suffixes=('_pop', '_growth'))
    
            # calculate new population per polygon based on the population in the former year, the population growth and the net migration
            merged_df['new_population_per_polygon_temp'] = merged_df['new_population_per_polygon'] + (merged_df['new_population_per_polygon'] * merged_df['population_growth']) + (merged_df['new_population_per_polygon'] * merged_df['y_pred'])
            merged_df.rename(columns={'new_population_per_polygon': 'population_t-1'}, inplace=True)
            merged_df.rename(columns={'new_population_per_polygon_temp': 'new_population_per_polygon'}, inplace=True)
            merged_df.to_csv(os.path.join(out_dir_PROJ, 'population_for_{}_exgeo.csv'.format(proj_year)))

        return merged_df 

            
        
            


