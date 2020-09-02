import pandas as pd
import geopandas as gpd
import numpy as np
import os, sys
from conflict_model import utils

def filter_conflict_properties(gdf, config):
    """filters conflict database according to certain conflict properties such as number of casualties, type of violence or country.

    Arguments:
        gdf {geodataframe}: geodataframe containing entries with conflicts
        config {configuration}: parsed configuration settings

    Returns:
        geodataframe: geodataframe containing filtered entries
    """    
    
    selection_criteria = {'best': config.getint('conflict', 'min_nr_casualties'),
                          'type_of_violence': (config.get('conflict', 'type_of_violence')).rsplit(',')}
    
    print('filtering on conflict properties...')
    
    for key in selection_criteria:

        if selection_criteria[key] == '':
            print('...passing key', key, 'as it is empty')
            pass

        elif key == 'best':
            print('...filtering key', key, 'with lower value', selection_criteria[key])
            gdf = gdf.loc[(gdf[key] >= selection_criteria[key])]

        else:
            print('...filtering key', key, 'with value(s)', selection_criteria[key])
            gdf = gdf.loc[(gdf[key].isin(selection_criteria[key]))]

    return gdf

def select_period(gdf, config):
    """Reducing the geodataframe to those entries falling into a specified time period.

    Arguments:
        gdf {geodataframe}: geodataframe containing entries with conflicts
        config {configuration}: parsed configuration settings

    Returns:
        geodataframe: geodataframe containing filtered entries
    """    

    t0 = config.getint('settings', 'y_start')
    t1 = config.getint('settings', 'y_end')
    
    print('focussing on period between', t0, 'and', t1)
    print('')
    
    gdf = gdf.loc[(gdf.year >= t0) & (gdf.year <= t1)]
    
    return gdf

def clip_to_extent(gdf, config):
    """As the original conflict data has global extent, this function clips the database to those entries which have occured on a specified continent.

    Arguments:
        gdf {geodataframe}: geodataframe containing entries with conflicts
        config {configuration}: parsed configuration settings

    Returns:
        geodataframe: geodataframe containing filtered entries
        geodataframe: geodataframe containing country polygons of selected continent
    """    
    
    shp_fo = os.path.join(os.path.abspath(config.get('general', 'input_dir')), 
                          config.get('extent', 'shp'))
    
    print('reading extent and spatial aggregation level from file {}'.format(shp_fo))
    extent_gdf = gpd.read_file(shp_fo)
    print('...DONE' + os.linesep)

    print('fixing invalid geometries')
    extent_gdf.geometry = extent_gdf.buffer(0)
    print('...DONE' + os.linesep)

    print('clipping clipping conflict dataset to extent')    
    gdf = gpd.clip(gdf, extent_gdf)
    print('...DONE' + os.linesep)
    
    return gdf, extent_gdf

def climate_zoning(gdf, extent_gdf, config):
    """[summary]

    Args:
        gdf ([type]): [description]
        extent_gdf ([type]): [description]
        config ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
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

        print('clipping conflicts to climate zones {}'.format(look_up_classes))
        gdf = gpd.clip(gdf, KG_gdf.buffer(0))
        print('...DONE' + os.linesep)

        print('clipping polygons to climate zones {}'.format(look_up_classes))
        extent_active_polys_gdf = gpd.clip(extent_gdf, KG_gdf.buffer(0))
        print('...DONE' + os.linesep)

    elif config.get('climate', 'zones') == 'None':

        gdf = gdf.copy()
        extent_active_polys_gdf = extent_gdf.copy()

    else:

        raise ValueError('no supported climate zone specified - either specify abbreviations of Koeppen-Geiger zones for selection or None for no selection')

    global_df = utils.global_ID_geom_info(extent_active_polys_gdf)

    return gdf, extent_active_polys_gdf, global_df

def select(config, plotting=False):
    """Filtering the original global conflict dataset based on a) conflict properties, b) time period, c) continent, and d) climate zone.

    Arguments:
        config {configuration}: parsed configuration settings

    Keyword Arguments:
        plotting {bool}: whether or not to plot the resulting selection (default: False)

    Returns:
        geodataframe: geodataframe containing filtered entries
        geodataframe: geodataframe containing country polygons of selected continent
    """    

    gdf = utils.get_geodataframe(config)

    gdf = filter_conflict_properties(gdf, config)

    gdf = select_period(gdf, config)

    gdf, extent_gdf = clip_to_extent(gdf, config)

    gdf, extent_active_polys_gdf, global_df = climate_zoning(gdf, extent_gdf, config)

    # if specified, plot the result
    if plotting:
        print('plotting result' + os.linesep)
        ax = gdf.plot(figsize=(10,5), legend=True, label='PRIO/UCDP events')
        extent_gdf.boundary.plot(ax=ax, color='0.5', linestyle=':')
        plt.legend()
        ax.set_xlim(extent_gdf.total_bounds[0]-1, extent_gdf.total_bounds[2]+1)
        ax.set_ylim(extent_gdf.total_bounds[1]-1, extent_gdf.total_bounds[3]+1)

    return gdf, extent_gdf, extent_active_polys_gdf, global_df