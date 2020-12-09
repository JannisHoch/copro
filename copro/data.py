from copro import conflict, variables
import numpy as np
import xarray as xr
import pandas as pd
import os, sys


def initiate_XY_data(config):
    """Initiates an empty dictionary to contain the XY-data for each polygon. 
    By default, the first column is for the polygon ID, the second for polygon geometry, and the last for binary conflict data (i.e. the Y-data).
    Every column in between corresponds to the variables provided in the cfg-file (i.e. the X-data).

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: emtpy dictionary to be filled, containing keys for each variable (X), binary conflict data (Y) plus meta-data.
    """

    XY = {}
    XY['poly_ID'] = pd.Series()
    XY['poly_geometry'] = pd.Series()
    for key in config.items('data'):
        XY[str(key[0])] = pd.Series(dtype=float)
    XY['conflict'] = pd.Series(dtype=int)

    if config.getboolean('general', 'verbose'): print('{}'.format(XY) + os.linesep)

    return XY

def initiate_X_data(config):
    """Initiates an empty dictionary to contain the X-data for each polygon. 
    By default, the first column is for the polygon ID and the second for polygon geometry.
    All remaining columns correspond to the variables provided in the cfg-file (i.e. the X-data).

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: emtpy dictionary to be filled, containing keys for each variable (X) plus meta-data.
    """    
    
    X = {}
    X['poly_ID'] = pd.Series()
    X['poly_geometry'] = pd.Series()
    for key in config.items('data'):
        X[str(key[0])] = pd.Series(dtype=float)

    if config.getboolean('general', 'verbose'): print('{}'.format(X) + os.linesep)

    return X

def fill_XY(XY, config, root_dir, conflict_gdf, polygon_gdf):
    """Fills the XY-dictionary with data for each variable and conflict for each polygon for each simulation year. 
    The number of rows should therefore equal to number simulation years times number of polygons.
    At end of last simulation year, the dictionary is converted to a numpy-array.

    Args:
        XY (dict): initiated, i.e. empty, XY-dictionary
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.

    Raises:
        Warning: a warning is raised if the datetime-format of the netCDF-file does not match conventions and/or supported formats.

    Returns:
        array: filled array containing the variable values (X) and binary conflict data (Y) plus meta-data.
    """    

    print('INFO: reading data for period from', str(config.getint('settings', 'y_start')), 'to', str(config.getint('settings', 'y_end')))

    # go through all simulation years as specified in config-file
    model_period = np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end') + 1, 1)
    for sim_year in model_period:

        print('INFO: entering year {}'.format(sim_year))

        # go through all keys in dictionary
        for key, value in XY.items(): 

            if key == 'conflict':
            
                data_series = value
                data_list = conflict.conflict_in_year_bool(conflict_gdf, polygon_gdf, config, sim_year)
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

                nc_ds = xr.open_dataset(os.path.join(root_dir, config.get('general', 'input_dir'), config.get('data', key)))
                
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
                    raise Warning('WARNING: this nc-file does have a different dtype for the time variable than currently supported: {}'.format(nc_fo))

    print('INFO: all data read')
    
    return pd.DataFrame.from_dict(XY).to_numpy()

def split_XY_data(XY, config):
    """Separates the XY-array into array containing information about variable values (X-array) and conflict data (Y-array).
    Thereby, the X-array also contains the information about unique identifier and polygon geometry.

    Args:
        XY (array): array containing variable values and conflict data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        arrays: two separate arrays, the X-array and Y-array
    """    

    XY = pd.DataFrame(XY)
    if config.getboolean('general', 'verbose'): print('DEBUG: number of data points including missing values:', len(XY))

    XY = XY.dropna()
    if config.getboolean('general', 'verbose'): print('DEBUG: number of data points excluding missing values:', len(XY))

    XY = XY.to_numpy()
    X = XY[:, :-1] # since conflict is the last column, we know that all previous columns must be variable values
    Y = XY[:, -1]
    Y = Y.astype(int)

    if config.getboolean('general', 'verbose'): 
        fraction_Y_1 = 100*len(np.where(Y != 0)[0])/len(Y)
        print('DEBUG: a fraction of {} percent in the data corresponds to conflicts.'.format(round(fraction_Y_1, 2)))

    return X, Y

def neighboring_polys(config, extent_gdf, identifier='watprovID'):

    # initialise empty dataframe
    df = pd.DataFrame()

    # go through each polygon aka water province
    for i in range(len(extent_gdf)):
        if config.getboolean('general', 'verbose'): print('DEBUG: finding touching neighbours for identifier {} {}'.format(identifier, extent_gdf[identifier].iloc[i]))
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