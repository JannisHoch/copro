from conflict_model import conflict, variables
import numpy as np
import xarray as xr
import pandas as pd
import os, sys


def initiate_XY_data(config):
    """[summary]

    Args:
        config ([type]): [description]

    Returns:
        [type]: [description]
    """    

    XY = {}
    XY['poly_ID'] = pd.Series()
    XY['poly_geometry'] = pd.Series()
    for key in config.items('env_vars'):
        XY[str(key[0])] = pd.Series(dtype=float)
    XY['conflict'] = pd.Series(dtype=int)

    if config.getboolean('general', 'verbose'): print('{}'.format(XY) + os.linesep)

    return XY

def fill_XY(XY, config, conflict_gdf, extent_active_polys_gdf):
    """[summary]

    Args:
        XY ([type]): [description]
        config ([type]): [description]
        conflict_gdf ([type]): [description]
        extent_active_polys_gdf ([type]): [description]

    Raises:
        Warning: [description]

    Returns:
        [type]: [description]
    """    

    if config.getboolean('general', 'verbose'): print('reading data for period from', str(config.getint('settings', 'y_start')), 'to', str(config.getint('settings', 'y_end')) + os.linesep)

    # go through all simulation years as specified in config-file
    for sim_year in np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end'), 1):

        if config.getboolean('general', 'verbose'): print(os.linesep + 'entering year {}'.format(sim_year) + os.linesep)

        # go through all keys in dictionary
        for key, value in XY.items(): 

            if key == 'conflict':
            
                data_series = value
                data_list = conflict.conflict_in_year_bool(conflict_gdf, extent_active_polys_gdf, config, sim_year)
                data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                XY[key] = data_series

            elif key == 'poly_ID':
            
                data_series = value
                data_list = conflict.get_poly_ID(extent_active_polys_gdf)
                data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                XY[key] = data_series

            elif key == 'poly_geometry':
            
                data_series = value
                data_list = conflict.get_poly_geometry(extent_active_polys_gdf)
                data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                XY[key] = data_series

            else:
            
                nc_ds = xr.open_dataset(os.path.join(config.get('general', 'input_dir'), config.get('env_vars', key)))
                
                if (np.dtype(nc_ds.time) == np.float32) or (np.dtype(nc_ds.time) == np.float64):
                    data_series = value
                    data_list = variables.nc_with_float_timestamp(extent_active_polys_gdf, config, key, sim_year)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series
                    
                elif np.dtype(nc_ds.time) == 'datetime64[ns]':
                    data_series = value
                    data_list = variables.nc_with_continous_datetime_timestamp(extent_active_polys_gdf, config, key, sim_year)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series
                    
                else:
                    raise Warning('this nc-file does have a different dtype for the time variable than currently supported: {}'.format(nc_fo))

    if config.getboolean('general', 'verbose'): print('...reading data DONE' + os.linesep)
    
    return XY

def split_XY_data(XY, config):
    """[summary]

    Args:
        XY ([type]): [description]

    Returns:
        [type]: [description]
    """    

    XY = pd.DataFrame.from_dict(XY)
    if config.getboolean('general', 'verbose'): print('number of data points including missing values:', len(XY))

    XY = XY.dropna()
    if config.getboolean('general', 'verbose'): print('number of data points excluding missing values:', len(XY))

    X = XY.to_numpy()[:, :-1] # since conflict is the last column, we know that all previous columns must be variable values
    Y = XY.conflict.astype(int).to_numpy()

    if config.getboolean('general', 'verbose'): 
        fraction_Y_1 = 100*len(np.where(Y != 0)[0])/len(Y)
        print('from this, {0} points are equal to 1, i.e. represent conflict occurence. This is a fraction of {1} percent.'.format(len(np.where(Y != 0)[0]), round(fraction_Y_1, 2)))

    return X, Y