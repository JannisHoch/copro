import pandas as pd
import geopandas as gpd
import numpy as np
import os, sys
from copro import utils

def filter_conflict_properties(gdf, config):
    """Filters conflict database according to certain conflict properties such as number of casualties, type of violence or country.

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with conflicts.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
    """    
    
    selection_criteria = {'best': config.getint('conflict', 'min_nr_casualties'),
                          'type_of_violence': (config.get('conflict', 'type_of_violence')).rsplit(',')}
    
    print('INFO: filtering on conflict properties.')
    
    for key in selection_criteria:

        if selection_criteria[key] == '':
            if config.getboolean('general', 'verbose'): print('DEBUG: passing key', key, 'as it is empty')
            pass

        elif key == 'best':
            if config.getboolean('general', 'verbose'): print('DEBUG: filtering key', key, 'with lower value', selection_criteria[key])
            gdf = gdf.loc[(gdf[key] >= selection_criteria[key])]

        else:
            if config.getboolean('general', 'verbose'): print('DEBUG: filtering key', key, 'with value(s)', selection_criteria[key])
            gdf = gdf.loc[(gdf[key].isin(selection_criteria[key]))]

    return gdf

def select_period(gdf, config):
    """Reducing the geo-dataframe to those entries falling into a specified time period.

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with conflicts.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
    """    

    t0 = config.getint('settings', 'y_start')
    t1 = config.getint('settings', 'y_end')
    
    if config.getboolean('general', 'verbose'): print('DEBUG: focussing on period between {} and {}'.format(t0, t1))
    
    gdf = gdf.loc[(gdf.year >= t0) & (gdf.year <= t1)]
    
    return gdf

def clip_to_extent(gdf, config):
    """As the original conflict data has global extent, this function clips the database to those entries which have occured on a specified continent.

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with conflicts.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
        geo-dataframe: geo-dataframe containing country polygons of selected continent.
    """    
    
    shp_fo = os.path.join(os.path.abspath(config.get('general', 'input_dir')), 
                          config.get('extent', 'shp'))
    
    print('INFO: reading extent and spatial aggregation level from file {}'.format(shp_fo))
    extent_gdf = gpd.read_file(shp_fo)

    print('INFO: fixing invalid geometries')
    extent_gdf.geometry = extent_gdf.buffer(0)

    print('INFO: clipping clipping conflict dataset to extent')    
    gdf = gpd.clip(gdf, extent_gdf)
    
    return gdf, extent_gdf

def climate_zoning(gdf, extent_gdf, config):
    """This function allows for selecting only those conflicts and polygons falling in specified climate zones.

    Args:
        gdf (geo-dataframe): geo-dataframe containing conflict data.
        extent_gdf (geo-dataframe): all polygons of study area.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Raises:
        ValueError: raised if a climate zone is specified which is not found in Koeppen-Geiger classification.

    Returns:
        geo-dataframe: conflict data clipped to climate zones.
        geo-dataframe: polygons of study area clipped to climate zones.
        dataframe: global look-up dataframe linking polygon ID with geometry information.
    """
    
    Koeppen_Geiger_fo = os.path.join(os.path.abspath(config.get('general', 'input_dir')),
                                     config.get('climate', 'shp')) 
    
    code2class_fo = os.path.join(os.path.abspath(config.get('general', 'input_dir')),
                                 config.get('climate', 'code2class'))
    
    KG_gdf = gpd.read_file(Koeppen_Geiger_fo)
    code2class = pd.read_csv(code2class_fo, sep='\t')
    
    if config.get('climate', 'zones') != 'None':

        look_up_classes = config.get('climate', 'zones').rsplit(',')

        code_nrs = []
        for entry in look_up_classes:
            code_nr = int(code2class['code'].loc[code2class['class'] == entry])
            code_nrs.append(code_nr)
    
        KG_gdf = KG_gdf.loc[KG_gdf['GRIDCODE'].isin(code_nrs)]
        
        if KG_gdf.crs != 'EPSG:4326':
            KG_gdf = KG_gdf.to_crs('EPSG:4326')

        if config.getboolean('general', 'verbose'): print('DEBUG: clipping conflicts to climate zones {}'.format(look_up_classes))
        gdf = gpd.clip(gdf, KG_gdf.buffer(0))

        if config.getboolean('general', 'verbose'): print('DEBUG: clipping polygons to climate zones {}'.format(look_up_classes))
        polygon_gdf = gpd.clip(extent_gdf, KG_gdf.buffer(0))

    elif config.get('climate', 'zones') == 'None':

        gdf = gdf.copy()
        polygon_gdf = extent_gdf.copy()

    else:

        raise ValueError('no supported climate zone specified - either specify abbreviations of Koeppen-Geiger zones for selection or None for no selection')

    global_df = utils.global_ID_geom_info(polygon_gdf)

    return gdf, polygon_gdf, global_df

def select(config, out_dir):
    """Main function performing the selection steps.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        geo-dataframe: remaining conflict data after selection process.
        geo-dataframe: all polygons of the study area.
        geo-dataframe: remaining polygons after selection process.
        dataframe: global look-up dataframe linking polygon ID with geometry information.
    """  

    gdf = utils.get_geodataframe(config)

    gdf = filter_conflict_properties(gdf, config)

    gdf = select_period(gdf, config)

    gdf, extent_gdf = clip_to_extent(gdf, config)

    gdf, polygon_gdf, global_df = climate_zoning(gdf, extent_gdf, config)

    gdf.to_file(os.path.join(out_dir, 'selected_conflicts.shp'), crs='EPSG:4326')
    polygon_gdf.to_file(os.path.join(out_dir, 'selected_polygons.shp'), crs='EPSG:4326')

    return gdf, extent_gdf, polygon_gdf, global_df