from copro import conflict, variables, evaluation
import click
import numpy as np
import xarray as xr
import pandas as pd
import os, sys


def initiate_XY_data(config):
    """Initiates an empty dictionary to contain the XY-data for each polygon, ie. both sample data and target data. 
    This is needed for the reference run.
    By default, the first column is for the polygon ID, the second for polygon geometry.
    The antepenultimate column is for boolean information about conflict at t-1 while the penultimate column is for boolean information about conflict at t-1 in neighboring polygons.
    The last column is for binary conflict data at t (i.e. the target data).
    
    Every column in between corresponds to the variables provided in the cfg-file.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: emtpy dictionary to be filled, containing keys for each variable (X), binary conflict data (Y) plus meta-data.
    """

    # Initialize dictionary
    # some entries are set by default, besides the ones corresponding to input data variables
    XY = {}
    XY['poly_ID'] = pd.Series()
    XY['poly_geometry'] = pd.Series()
    for key in config.items('data'):
        XY[str(key[0])] = pd.Series(dtype=float)
    XY['conflict_t_min_1'] = pd.Series(dtype=bool)
    XY['conflict_t_min_1_nb'] = pd.Series(dtype=float)
    XY['conflict'] = pd.Series(dtype=bool)

    if config.getboolean('general', 'verbose'): 
        click.echo('DEBUG: the columns in the sample matrix used are:')
        for key in XY:
            click.echo('...{}'.format(key))

    return XY

def initiate_X_data(config):
    """Initiates an empty dictionary to contain the X-data for each polygon, ie. only sample data. 
    This is needed for each time step of each projection run.
    By default, the first column is for the polygon ID and the second for polygon geometry.
    The penultimate column is for boolean information about conflict at t-1 while the last column is for boolean information about conflict at t-1 in neighboring polygons.
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
    X['conflict_t_min_1'] = pd.Series(dtype=bool)
    X['conflict_t_min_1_nb'] = pd.Series(dtype=float)

    if config.getboolean('general', 'verbose'): 
        click.echo('DEBUG: the columns in the sample matrix used are:')
        for key in X:
            click.echo('...{}'.format(key))

    return X

def fill_XY(XY, config, root_dir, conflict_data, polygon_gdf, out_dir):
    """Fills the (XY-)dictionary with data for each variable and conflict for each polygon for each simulation year. 
    The number of rows should therefore equal to number simulation years times number of polygons.
    At end of last simulation year, the dictionary is converted to a numpy-array.

    Args:
        XY (dict): initiated, i.e. empty, XY-dictionary
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.
        conflict_data (geo-dataframe): geo-dataframe containing the selected conflicts.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        out_dir (path): path to output folder.

    Raises:
        Warning: raised if the datetime-format of the netCDF-file does not match conventions and/or supported formats.

    Returns:
        array: filled array containing the variable values (X) and binary conflict data (Y) plus meta-data.
    """    

    # go through all simulation years as specified in config-file
    model_period = np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end') + 1, 1)
    click.echo('INFO: reading data for period from {} to {}'.format(model_period[0], model_period[-1]))

    neighboring_matrix = neighboring_polys(config, polygon_gdf)

    for (sim_year, i) in zip(model_period, range(len(model_period))):

        if i == 0:

            click.echo('INFO: skipping first year {} to start up model'.format(sim_year))

        else:

            click.echo('INFO: entering year {}'.format(sim_year))

            # go through all keys in dictionary
            for key, value in XY.items(): 

                if key == 'conflict':
                
                    data_series = value
                    data_list = conflict.conflict_in_year_bool(config, conflict_data, polygon_gdf, sim_year, out_dir)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series

                elif key == 'conflict_t_min_1':

                    data_series = value
                    data_list = conflict.conflict_in_previous_year(config, conflict_data, polygon_gdf, sim_year)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series

                elif key == 'conflict_t_min_1_nb':

                    data_series = value
                    data_list = conflict.conflict_in_previous_year(config, conflict_data, polygon_gdf, sim_year, check_neighbors=True, neighboring_matrix=neighboring_matrix)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series

                elif key == 'poly_ID':
                
                    data_series = value
                    data_list = conflict.get_poly_ID(polygon_gdf)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series

                elif key == 'poly_geometry':
                
                    data_series = value
                    data_list = conflict.get_poly_geometry(polygon_gdf, config)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series

                else:

                    nc_ds = xr.open_dataset(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key)).rsplit(',')[0])
                    
                    if (np.dtype(nc_ds.time) == np.float32) or (np.dtype(nc_ds.time) == np.float64):
                        data_series = value
                        data_list = variables.nc_with_float_timestamp(polygon_gdf, config, root_dir, key, sim_year)
                        data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                        XY[key] = data_series
                        
                    elif np.dtype(nc_ds.time) == 'datetime64[ns]':
                        data_series = value
                        data_list = variables.nc_with_continous_datetime_timestamp(polygon_gdf, config, root_dir, key, sim_year)
                        data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                        XY[key] = data_series
                        
                    else:
                        raise Warning('WARNING: this nc-file does have a different dtype for the time variable than currently supported: {}'.format(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key))))

            if config.getboolean('general', 'verbose'): click.echo('DEBUG: all data read')

    df_out = pd.DataFrame.from_dict(XY)
    
    return df_out.to_numpy()

def fill_X_sample(X, config, root_dir, polygon_gdf, proj_year):
    """Fills the X-dictionary with the data sample data besides any conflict-related data for each polygon and each year.
    Used during the projection runs as the sample and conflict data need to be treated separately there.

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
            data_list = conflict.get_poly_ID(polygon_gdf)
            data_series = data_series.append(pd.Series(data_list), ignore_index=True)
            X[key] = data_series

        elif key == 'poly_geometry':
        
            data_series = value
            data_list = conflict.get_poly_geometry(polygon_gdf, config)
            data_series = data_series.append(pd.Series(data_list), ignore_index=True)
            X[key] = data_series

        else:

            if (key != 'conflict_t_min_1') and (key != 'conflict_t_min_1_nb'):

                nc_ds = xr.open_dataset(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key)).rsplit(',')[0])
                
                if (np.dtype(nc_ds.time) == np.float32) or (np.dtype(nc_ds.time) == np.float64):
                    data_series = value
                    data_list = variables.nc_with_float_timestamp(polygon_gdf, config, root_dir, key, proj_year)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    X[key] = data_series
                    
                elif np.dtype(nc_ds.time) == 'datetime64[ns]':
                    data_series = value
                    data_list = variables.nc_with_continous_datetime_timestamp(polygon_gdf, config, root_dir, key, proj_year)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    X[key] = data_series
                    
                else:
                    raise Warning('WARNING: this nc-file does have a different dtype for the time variable than currently supported: {}'.format(nc_fo))

    return X

def fill_X_conflict(X, config, conflict_data, polygon_gdf):
    """Fills the X-dictionary with the conflict data for each polygon and each year.
    Used during the projection runs as the sample and conflict data need to be treated separately there.

    Args:
        X (dict): dictionary containing keys to be sampled.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        conflict_data (dataframe): dataframe containing all polygons with conflict.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.

    Returns:
        dict: dictionary containing sample and conflict values.
    """    

    # determine all neighbours for each polygon
    neighboring_matrix = neighboring_polys(config, polygon_gdf)

    # go through all keys in dictionary
    for key, value in X.items(): 

        if key == 'conflict_t_min_1':

            data_series = value
            data_list = conflict.read_projected_conflict(polygon_gdf, conflict_data)
            data_series = data_series.append(pd.Series(data_list), ignore_index=True)
            X[key] = data_series

        elif key == 'conflict_t_min_1_nb':

            data_series = value
            data_list = conflict.read_projected_conflict(polygon_gdf, conflict_data, check_neighbors=True, neighboring_matrix=neighboring_matrix)
            data_series = data_series.append(pd.Series(data_list), ignore_index=True)
            X[key] = data_series

        else:

            pass

    if config.getboolean('general', 'verbose'): click.echo('DEBUG: all data read')

    return X

def split_XY_data(XY, config):
    """Separates the XY-array into array containing information about variable values (X-array or sample data) and conflict data (Y-array or target data).
    Thereby, the X-array also contains the information about unique identifier and polygon geometry.

    Args:
        XY (array): array containing variable values and conflict data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        arrays: two separate arrays, the X-array and Y-array.
    """    

    # convert array to dataframe for easier handling
    XY = pd.DataFrame(XY)
    if config.getboolean('general', 'verbose'): click.echo('DEBUG: number of data points including missing values: {}'.format(len(XY)))

    # fill all missing values with 0
    XY = XY.fillna(0)

    # convert dataframe back to array
    XY = XY.to_numpy()
    
    # get X data
    # since conflict is the last column, we know that all previous columns must be variable values
    X = XY[:, :-1] 
    # get Y data and convert to integer values
    Y = XY[:, -1]
    Y = Y.astype(int)

    if config.getboolean('general', 'verbose'): 
        fraction_Y_1 = 100*len(np.where(Y != 0)[0])/len(Y)
        click.echo('DEBUG: a fraction of {} percent in the data corresponds to conflicts.'.format(round(fraction_Y_1, 2)))

    return X, Y

def neighboring_polys(config, extent_gdf, identifier='watprovID'):
    """For each polygon, determines its neighboring polygons.
    As result, a (n x n) look-up dataframe is obtained containing, where n is number of polygons in extent_gdf.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        extent_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        identifier (str, optional): column name in extent_gdf to be used to identify neighbors. Defaults to 'watprovID'.

    Returns:
        dataframe: look-up dataframe containing True/False statement per polygon for all other polygons.
    """    

    if config.getboolean('general', 'verbose'): click.echo('DEBUG: determining matrix with neighboring polygons')

    # initialise empty dataframe
    df = pd.DataFrame()

    # go through each polygon aka water province
    for i in range(len(extent_gdf)):
        # get geometry of current polygon
        wp = extent_gdf.geometry.iloc[i]
        # check which polygons in geodataframe (i.e. all water provinces) touch the current polygon
        # also create a dataframe from result (boolean)
        # the transpose is needed to easier append
        df_temp = pd.DataFrame(extent_gdf.geometry.touches(wp), columns=[extent_gdf[identifier].iloc[i]]).T
        # append the dataframe
        df = df.append(df_temp)

    # replace generic indices with actual water province IDs
    df.set_index(extent_gdf[identifier], inplace=True)

    # replace generic columns with actual water province IDs
    df.columns = extent_gdf[identifier].values

    return df

def find_neighbors(ID, neighboring_matrix):
    """Filters all polygons which are actually neighbors to given polygon.

    Args:
        ID (int): ID of specific polygon under consideration.
        neighboring_matrix (dataframe): output from neighboring_polys().

    Returns:
        dataframe: dataframe containig IDs of all polygons that are actual neighbors.
    """    

    # locaties entry for polygon under consideration
    neighbours = neighboring_matrix.loc[neighboring_matrix.index == ID].T
    
    # filters all actual neighbors defined as neighboring polygons with True statement
    actual_neighbours = neighbours.loc[neighbours[ID] == True].index.values

    return actual_neighbours