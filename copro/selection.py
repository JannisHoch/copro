import pandas as pd
import geopandas as gpd
import os
from copro import utils


def select_period(gdf, config):
    """Reducing the geo-dataframe to those entries falling into a specified time period.

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with migration.
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

def define_extent(gdf, config, root_dir):

  # get path to file with polygons for which analysis is carried out
    shp_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('extent', 'shp'))
    
    # read file
    if config.getboolean('general', 'verbose'): print('DEBUG: reading extent and spatial aggregation level from file {}'.format(shp_fo))
    extent_gdf = gpd.read_file(shp_fo)

    # fixing invalid geometries
    if config.getboolean('general', 'verbose'): print('DEBUG: fixing invalid geometries')
    extent_gdf.geometry = extent_gdf.buffer(0)

    if (gdf.crs == extent_gdf.crs):
        print("DEBUG: Both layers are in the same crs")

    # clip the migration dataframe to the specified polygons
    
    #if config.getboolean('general', 'verbose'):
        #print('DEBUG: clipping migration dataset to extent')
    
    # gdf.intersects(extent_gdf.unary_union) 

    return extent_gdf

# def clip_to_extent(gdf, config, root_dir):
   
    """As the original copro worked with a global extent, this function clips the database to those entries which have occured on a specified continent. 

    Args:
        gdf (geo-dataframe): geo-dataframe containing entries with migration.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
        geo-dataframe: geo-dataframe containing country polygons of selected continent.
    """    

    # get path to file with polygons for which analysis is carried out
    #shp_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('extent', 'shp'))
    
    # read file
    #if config.getboolean('general', 'verbose'): print('DEBUG: reading extent and spatial aggregation level from file {}'.format(shp_fo))
    #extent_gdf = gpd.read_file(shp_fo)

    # fixing invalid geometries
    #if config.getboolean('general', 'verbose'): print('DEBUG: fixing invalid geometries')
    #extent_gdf.geometry = extent_gdf.buffer(0)

    #if (gdf.crs == extent_gdf.crs):
    #print("DEBUG: Both layers are in the same crs")

    # clip the migration dataframe to the specified polygons
    
    #if config.getboolean('general', 'verbose'):
        #print('DEBUG: clipping migration dataset to extent')
    
    # gdf.intersects(extent_gdf.unary_union) 

    #if isinstance(gdf, gpd.GeoDataFrame):
        #print("Yes, it is a GeoDataFrame")
    # else:
        #print("No, it is not a GeoDataFrame")

    #return gdf, extent_gdf

def climate_zoning(gdf, config, root_dir): 
    """This function allows for selecting only those migration data and polygons falling in specified climate zones.
    Also, a global dataframe is returned containing the IDs and geometry of all polygons after selection procedure.
    This can be used to add geometry information to model output based on common ID.

    Args:
        gdf (geo-dataframe): geo-dataframe containing migration data.
        extent_gdf (geo-dataframe): all polygons of study area.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.

    Returns:
        geo-dataframe: migration data clipped to climate zones.
        geo-dataframe: polygons of study area clipped to climate zones.
    """

    # load file with extents of climate zones
    Koeppen_Geiger_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('climate', 'shp'))
    KG_gdf = gpd.read_file(Koeppen_Geiger_fo)
    # load file to look-up climate zone names with codes in shp-file
    code2class_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('climate', 'code2class'))
    code2class = pd.read_csv(code2class_fo, sep='\t')
    
    # if climate zones are specified...
    if config.get('climate', 'zones') !='':
        print('code condtinues5')
   
        # get all classes specified
        look_up_classes = config.get('climate', 'zones').rsplit(',')
        
        print('code condtinues6')

        # get the corresponding code per class
        code_nrs = []
        for entry in look_up_classes:
            code_nr = int(code2class['code'].loc[code2class['class'] == entry])
            code_nrs.append(code_nr)
    
        # get only those entries with retrieved codes
        KG_gdf = KG_gdf.loc[KG_gdf['GRIDCODE'].isin(code_nrs)]
        
        # make sure EPSG:4236 is used --> now changed into WGS84
        if KG_gdf.crs != 'WGS84':
            KG_gdf = KG_gdf.to_crs('WGS84')

        # clip the migration dataframe to the specified climate zones
        if config.getboolean('general', 'verbose'): print('DEBUG: clipping migration to climate zones {}'.format(look_up_classes))
        gdf = gpd.clip(gdf, KG_gdf.buffer(0))

        # clip the studied polygons to the specified climate zones
        if config.getboolean('general', 'verbose'): print('DEBUG: clipping polygons to climate zones {}'.format(look_up_classes))
        polygon_gdf = gpd.clip(gdf, KG_gdf.buffer(0))

    # if not, nothing needs to be done besides aligning names
    else:

        polygon_gdf = gdf.copy()

    return gdf, polygon_gdf

def select(config, out_dir, root_dir):
    """Main function performing the selection procedure.
    Also stores the selected migration data and polygons to output directory.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder.
        root_dir (str): path to location of cfg-file.

    Returns:
        geo-dataframe: remaining migration data after selection process.
        geo-dataframe: all polygons of the study area.
        geo-dataframe: remaining polygons after selection process.
        dataframe: global look-up dataframe linking polygon ID with geometry information.
    """  

    # get the migration data
    gdf = utils.get_geodataframe(config, root_dir)

    # filter based on migration properties --> THIS CAN MOST LIKELY BE DELETED
    # gdf = filter_migration_properties(gdf, config)

    # selected migration falling in a specified time period
    gdf = select_period(gdf, config)

    # clip migration to a spatial extent defined as polygons
    #gdf, extent_gdf = clip_to_extent(gdf, config, root_dir)

    # extent_gdf = define_extent(gdf, config, root_dir)

    # clip migration and polygons to specified climate zones
    gdf, polygon_gdf = climate_zoning(gdf, config, root_dir)

    extent_gdf = define_extent(gdf, config, root_dir)

    # get a dataframe containing the ID and geometry of all polygons after selecting for climate zones
    global_df = utils.global_ID_geom_info(polygon_gdf)

    # save migration data and polygon to shp-file
    gdf.to_file(os.path.join(out_dir, 'selected_migration.geojson'), driver='GeoJSON', crs='WGS84') #change projection
    polygon_gdf.to_file(os.path.join(out_dir, 'selected_polygons.geojson'), driver='GeoJSON', crs='WGS84') #change projection

    return gdf, polygon_gdf, global_df