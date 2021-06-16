import pandas as pd
import geopandas as gpd
import os
from copro import utils

def filter_conflict_properties(gdf, config):
    """Filters conflict database according to certain conflict properties such as number of casualties, type of violence or country.

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with conflicts.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
    """    
    
    # create dictionary with all selection criteria
    selection_criteria = {'best': config.getint('conflict', 'min_nr_casualties'),
                          'type_of_violence': (config.get('conflict', 'type_of_violence')).rsplit(',')}
    
    print('INFO: filtering based on conflict properties.')
    
    # go through all criteria
    for key in selection_criteria:

        # for criterion 'best' (i.e. best estimate of fatalities), select all entries above threshold
        if key == 'best':
            if selection_criteria[key] == '':
                pass
            else:
                if config.getboolean('general', 'verbose'): print('DEBUG: filtering key', key, 'with lower value', selection_criteria[key])
                gdf = gdf[gdf['best'] >= selection_criteria['best']]

        # for other criteria, select all entries matching the specified value(s) per criterion
        if key == 'type_of_violence':
            if selection_criteria[key] == '':
                pass
            else:
                if config.getboolean('general', 'verbose'): print('DEBUG: filtering key', key, 'with value(s)', selection_criteria[key])
                gdf = gdf[gdf[key].isin(selection_criteria[key])]

    return gdf

def select_period(gdf, config):
    """Reducing the geo-dataframe to those entries falling into a specified time period.

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with conflicts.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
    """    

    # get start and end year of model period
    t0 = config.getint('settings', 'y_start')
    t1 = config.getint('settings', 'y_end')
    
    # select those entries meeting the requirements
    if config.getboolean('general', 'verbose'): print('DEBUG: focussing on period between {} and {}'.format(t0, t1))
    gdf = gdf.loc[(gdf.year >= t0) & (gdf.year <= t1)]
    
    return gdf

def clip_to_extent(gdf, config, root_dir):
    """As the original conflict data has global extent, this function clips the database to those entries which have occured on a specified continent.

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with conflicts.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
        geo-dataframe: geo-dataframe containing country polygons of selected continent.
    """    

    # get path to file with polygons for which analysis is carried out
    shp_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('extent', 'shp'))
    
    # read file
    if config.getboolean('general', 'verbose'): print('DEBUG: reading extent and spatial aggregation level from file {}'.format(shp_fo))
    extent_gdf = gpd.read_file(shp_fo)

    # fixing invalid geometries
    if config.getboolean('general', 'verbose'): print('DEBUG: fixing invalid geometries')
    extent_gdf.geometry = extent_gdf.buffer(0)

    # clip the conflict dataframe to the specified polygons
    if config.getboolean('general', 'verbose'): print('DEBUG: clipping clipping conflict dataset to extent')    
    gdf = gpd.clip(gdf, extent_gdf)
    
    return gdf, extent_gdf

def climate_zoning(gdf, extent_gdf, config, root_dir):
    """This function allows for selecting only those conflicts and polygons falling in specified climate zones.
    Also, a global dataframe is returned containing the IDs and geometry of all polygons after selection procedure.
    This can be used to add geometry information to model output based on common ID.

    Args:
        gdf (geo-dataframe): geo-dataframe containing conflict data.
        extent_gdf (geo-dataframe): all polygons of study area.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.

    Returns:
        geo-dataframe: conflict data clipped to climate zones.
        geo-dataframe: polygons of study area clipped to climate zones.
    """

    # load file with extents of climate zones
    Koeppen_Geiger_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('climate', 'shp'))
    KG_gdf = gpd.read_file(Koeppen_Geiger_fo)
    # load file to look-up climate zone names with codes in shp-file
    code2class_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('climate', 'code2class'))
    code2class = pd.read_csv(code2class_fo, sep='\t')
    
    # if climate zones are specified...
    if config.get('climate', 'zones') != '':

        # get all classes specified
        look_up_classes = config.get('climate', 'zones').rsplit(',')

        # get the corresponding code per class
        code_nrs = []
        for entry in look_up_classes:
            code_nr = int(code2class['code'].loc[code2class['class'] == entry])
            code_nrs.append(code_nr)
    
        # get only those entries with retrieved codes
        KG_gdf = KG_gdf.loc[KG_gdf['GRIDCODE'].isin(code_nrs)]
        
        # make sure EPSG:4236 is used
        if KG_gdf.crs != 'EPSG:4326':
            KG_gdf = KG_gdf.to_crs('EPSG:4326')

        # clip the conflict dataframe to the specified climate zones
        if config.getboolean('general', 'verbose'): print('DEBUG: clipping conflicts to climate zones {}'.format(look_up_classes))
        gdf = gpd.clip(gdf, KG_gdf.buffer(0))

        # clip the studied polygons to the specified climate zones
        if config.getboolean('general', 'verbose'): print('DEBUG: clipping polygons to climate zones {}'.format(look_up_classes))
        polygon_gdf = gpd.clip(extent_gdf, KG_gdf.buffer(0))

    # if not, nothing needs to be done besides aligning names
    else:

        polygon_gdf = extent_gdf.copy()

    return gdf, polygon_gdf

def select(config, out_dir, root_dir):
    """Main function performing the selection procedure.
    Also stores the selected conflicts and polygons to output directory.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder.
        root_dir (str): path to location of cfg-file.

    Returns:
        geo-dataframe: remaining conflict data after selection process.
        geo-dataframe: all polygons of the study area.
        geo-dataframe: remaining polygons after selection process.
        dataframe: global look-up dataframe linking polygon ID with geometry information.
    """  

    # get the conflict data
    gdf = utils.get_geodataframe(config, root_dir)

    # filter based on conflict properties
    gdf = filter_conflict_properties(gdf, config)

    # selected conflicts falling in a specified time period
    gdf = select_period(gdf, config)

    # clip conflicts to a spatial extent defined as polygons
    gdf, extent_gdf = clip_to_extent(gdf, config, root_dir)

    # clip conflicts and polygons to specified climate zones
    gdf, polygon_gdf = climate_zoning(gdf, extent_gdf, config, root_dir)

    # get a dataframe containing the ID and geometry of all polygons after selecting for climate zones
    global_df = utils.global_ID_geom_info(polygon_gdf)

    # save conflict data and polygon to shp-file
    # TODO: save as geoJSON rather than shp
    gdf.to_file(os.path.join(out_dir, 'selected_conflicts.shp'), crs='EPSG:4326')
    polygon_gdf.to_file(os.path.join(out_dir, 'selected_polygons.shp'), crs='EPSG:4326')

    return gdf, polygon_gdf, global_df