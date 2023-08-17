from copro import migration, variables, evaluation, variables_3year_average
import click
import numpy as np
import xarray as xr
import pandas as pd
import os, sys


def initiate_XY_data(config):
    """Initiates an empty dictionary to contain the XY-data for each polygon, ie. both sample data and target data. 
    This is needed for the reference run.
    By default, the first column is for the polygon ID, the second for polygon geometry.
    # DELETE ALL FUNTIONS ON: 
        The antepenultimate column is for boolean information about conflict at t-1 while the penultimate column is for boolean information about conflict at t-1 in neighboring polygons.
    The last column is for binary conflict data at t (i.e. the target data).
    
    Every column in between corresponds to the variables provided in the cfg-file.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: emtpy dictionary to be filled, containing keys for each variable (X), migration data (Y) plus meta-data.
    """

    # Initialize dictionary
    # some entries are set by default, besides the ones corresponding to input data variables
    XY = {}
    XY['poly_ID'] = pd.Series()
    XY['poly_geometry'] = pd.Series()
    for key in config.items('data'):
        XY[str(key[0])] = pd.Series(dtype=float)
    XY['net_migration'] = pd.Series(dtype=int)

    if config.getboolean('general', 'verbose'): 
        click.echo('DEBUG: the columns in the sample matrix used are:')
        for key in XY:
            click.echo('...{}'.format(key))

    return XY

def initiate_X_data(config):
    """Initiates an empty dictionary to contain the X-data for each polygon, ie. only sample data. 
    This is needed for each time step of each projection run.
    By default, the first column is for the polygon ID and the second for polygon geometry.
    # DELETE CODE CONSIDERING: The penultimate column is for boolean information about conflict at t-1 while the last column is for boolean information about conflict at t-1 in neighboring polygons.
    All remaining columns correspond to the variables provided in the cfg-file.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: emtpy dictionary to be filled, containing keys for each variable (X) plus meta-data.
    """   

    # Initialize dictionary
    # some entries are set by default, besides the ones corresponding to input data variables
    X = {}
    X['poly_ID'] = pd.Series()
    X['poly_geometry'] = pd.Series()
    for key in config.items('data'):
        X[str(key[0])] = pd.Series(dtype=float)
    if config.getboolean('general', 'verbose'): 
        click.echo('DEBUG: the columns in the sample matrix used are:')
        for key in X:
            click.echo('...{}'.format(key))

    return X

def fill_XY(XY, config, root_dir, migration_data, polygon_gdf, out_dir):
    """Fills the (XY-)dictionary with data for each variable and migration for each polygon for each simulation year. 
    The number of rows should therefore equal to number simulation years times number of polygons.
    At end of last simulation year, the dictionary is converted to a numpy-array.

    Args:
        XY (dict): initiated, i.e. empty, XY-dictionary
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.
        migration (geo-dataframe): geo-dataframe containing the migration data.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        out_dir (path): path to output folder.

    Raises:
        Warning: raised if the datetime-format of the netCDF-file does not match conventions and/or supported formats.

    Returns:
        array: filled array containing the variable values (X) and migration data (Y) plus meta-data.
    """    

    # go through all simulation years as specified in config-file
    model_period = np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end') + 1, 1)
    click.echo('INFO: reading data for period from {} to {}'.format(model_period[0], model_period[-1])) 

    if config.getboolean('general', 'one_year_migration_average'):
        step = 1
    elif config.getboolean('general', 'three_year_migration_average'):
        step = 3
    elif config.getboolean('general', 'five_year_migration_average'):
        step = 5 
    else:
        raise ValueError('Invalid timestep configuration.')
    print('print step')
    print(step)

    for sim_year in range(config.getint('settings', 'y_start'), config.getint('settings', 'y_end') + 1, step):
        click.echo('INFO: entering year {}'.format(sim_year))

        # go through all keys in dictionary
        for key, value in XY.items(): 

                if key == 'net_migration':
                
                    data_series = value
                    if config.getboolean('general', 'one_year_migration_average'):
                        data_list = migration.migration_in_year_int (root_dir, config, migration_data, sim_year, out_dir)
                    elif config.getboolean('general', 'three_year_migration_average'):
                        data_list = migration.migration_in_three_years(root_dir, config, migration_data, sim_year, out_dir)
                    elif config.getboolean('general', 'three_year_migration_average'):
                        data_list = migration.migration_in_year_int (root_dir, config, migration_data, sim_year, out_dir)
                    data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                    XY[key] = data_series

                elif key == 'poly_ID':
                
                    data_series = value
                    data_list = migration.get_poly_ID(polygon_gdf)
                    data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                    XY[key] = data_series

                elif key == 'poly_geometry':
                
                    data_series = value
                    data_list = migration.get_poly_geometry(polygon_gdf, config)
                    data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                    XY[key] = data_series                        
                        
                else: 
                    file_path = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key)).rsplit(',')[0]
                    file_extension = os.path.splitext(file_path)[1]
                    
                    if file_extension == '.csv':
                        data_series = value 
                        if config.getboolean('general', 'one_year_migration_average'):
                            data_list = variables.csv_extract_value(polygon_gdf, config, root_dir, key, sim_year)
                        elif config.getboolean('general', 'three_year_migration_average'):
                            data_list = variables_3year_average.csv_extract_value(polygon_gdf, config, root_dir, key, sim_year)
                        data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                        XY[key] = data_series

                    elif file_extension == '.nc':                    
                        nc_ds = xr.open_dataset(file_path)

                        if (np.dtype(nc_ds.time) == np.float32) or (np.dtype(nc_ds.time) == np.float64):
                            data_series = value
                            data_list = variables.nc_with_float_timestamp(polygon_gdf, config, root_dir, key, sim_year)
                            data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                            XY[key] = data_series
                
                        elif np.dtype(nc_ds.time) == 'datetime64[ns]':
                            data_series = value
                            data_list = variables.nc_with_continous_datetime_timestamp(polygon_gdf, config, root_dir, key, sim_year)
                            data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                            XY[key] = data_series

                        else:
                            raise Warning('WARNING: this nc-file does have a different dtype for the time variable than currently supported: {}'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key))))
                    else:
                        raise ValueError('ERROR: the file extension of the input file is not supported: {}'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key))))

    # Sort the dictionary based on the 'poly_ID' key in the second element of the tuple columns

    sorted_XY = dict(sorted(XY.items(), key=lambda x: str(x[1][0])))

    # Delete the column named 'poly_ID' since somehow I cant fix it to het the right poly_ID in the correct row
    del sorted_XY['poly_ID']

    df_out = pd.DataFrame(sorted_XY)

    # Find the correct 'poly_ID' column and add it (again)
    poly_ID_column = next(col for col in df_out.columns if isinstance(df_out[col][0], tuple))

    # Insert a new column 'poly_ID' with the second element of the tuples
    df_out.insert(0, 'poly_ID', df_out[poly_ID_column].apply(lambda x: x[1] if isinstance(x, tuple) else x))

    # Find the correct 'poly_ID' column
    poly_ID_column = next(col for col in df_out.columns if isinstance(df_out[col][0], tuple))

    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, tuple)).all():
            df_out[col] = df_out[col].apply(lambda x: x[0]) 

    # make sure the order of the columns is correct for later analysis: 
    df_out = df_out[['poly_ID', 'poly_geometry'] + list(df_out.columns.drop(['poly_ID', 'poly_geometry', 'net_migration'])) + ['net_migration']]

    # Extract only the integer part from each tuple column
    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, tuple)).all():
           df_out[col] = df_out[col].apply(lambda x: x[0]) 
      
    if config.getboolean('general', 'verbose'):
        click.echo('DEBUG: all data read')


    # Drop the 'poly_geometry' column from the DataFrame temporarily for saving to CSV
    df_out_to_save = df_out.drop(columns=['poly_geometry'])

    # Save the temporary DataFrame to a CSV file
    df_out_to_save.to_csv(os.path.join(out_dir, 'DF_out_exgeometry.csv'), index=False, header=True)
    print('df_out_exgeometry.csv saved in the output folder')

    return df_out.to_numpy()


def fill_X_sample(X, config, root_dir, polygon_gdf, proj_year):
    """Fills the X-dictionary with the sample data besides the migration data for each polygon and each year.
    Used during the projection runs as the sample and migration data need to be treated separately there.

    Args:
        X (dict): dictionary containing keys to be sampled.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file of reference run.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        proj_year (int): year for which projection is made.

    Raises:
        Warning: raised if the datetime-format of the netCDF-file does not match conventions and/or supported formats.

    Returns:
        dict: dictionary containing sample values.
    """    

    if config.getboolean('general', 'verbose'): click.echo('DEBUG: reading sample data from files')

    # go through all keys in dictionary
    for key, value in X.items(): 

        if key == 'poly_ID':
        
            data_series = value
            data_list = migration.get_poly_ID(polygon_gdf)
            data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
            X[key] = data_series

        elif key == 'poly_geometry':
        
            data_series = value
            data_list = migration.get_poly_geometry(polygon_gdf, config)
            data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
            X[key] = data_series

        else: 
            file_path = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key)).rsplit(',')[0]
            file_extension = os.path.splitext(file_path)[1]
                    
            if file_extension == '.csv':
                data_series = value 
                data_list = variables.csv_extract_value(polygon_gdf, config, root_dir, key, proj_year)
                data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                X[key] = data_series

            elif file_extension == '.nc':                    
                nc_ds = xr.open_dataset(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key)).rsplit(',')[0])
                
                if (np.dtype(nc_ds.time) == np.float32) or (np.dtype(nc_ds.time) == np.float64):
                    data_series = value
                    data_list = variables.nc_with_float_timestamp(polygon_gdf, config, root_dir, key, proj_year)
                    data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                    X[key] = data_series
                    
                elif np.dtype(nc_ds.time) == 'datetime64[ns]':
                    data_series = value
                    data_list = variables.nc_with_continous_datetime_timestamp(polygon_gdf, config, root_dir, key, proj_year)
                    data_series = pd.concat([data_series, pd.Series(data_list)], axis=0, ignore_index=True)
                    X[key] = data_series
                    
                else:
                    raise Warning('WARNING: this nc-file does have a different dtype for the time variable than currently supported: {}'.format(nc_fo))

    # Delete the column named 'poly_ID'
    del X['poly_ID']

    #df_out = pd.DataFrame(sorted_X)
    df_out = pd.DataFrame(X)

    # Insert a new column 'poly_ID' with the second element of the tuples
    df_out.insert(0, 'poly_ID', df_out.iloc[:, 2].apply(lambda x: x[1]))

     # make sure the order of the columns is correct for later analysis: 
    df_out = df_out[['poly_ID', 'poly_geometry'] + list(df_out.columns.drop(['poly_ID', 'poly_geometry']))]

    # Extract only the integer part from each tuple column
    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, tuple)).all():
           df_out[col] = df_out[col].apply(lambda x: x[0]) 
      
    if config.getboolean('general', 'verbose'):
        click.echo('DEBUG: all X-prediction data read')

    X = df_out.set_index('poly_ID').to_dict(orient='index')
    
    return X

def split_XY_data(XY, config):
    """Separates the XY-array into array containing information about variable values (X-array or sample data) and migration data (Y-array or target data).
    Thereby, the X-array also contains the information about unique identifier and polygon geometry.

    Args:
        XY (array): array containing variable values and migration data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        arrays: two separate arrays, the X-array and Y-array.
    """    

    # convert array to dataframe for easier handling
    XY = pd.DataFrame(XY)

    # fill all missing values with 0
    XY = XY.fillna(0)

    # convert dataframe back to array
    XY = XY.to_numpy()
    
    # get X data
    # since migration is the last column, we know that all previous columns must be variable values
    X = XY[:, :-1] 

    # get Y data 
    Y = XY[:, -1]

    if config.getboolean('general', 'verbose'): 
        fraction_Y_1 = 100*len(np.where(Y != 0)[0])/len(Y)
        click.echo('DEBUG: a fraction of {} percent in the data corresponds to migration.'.format(round(fraction_Y_1, 2)))

    return X, Y
