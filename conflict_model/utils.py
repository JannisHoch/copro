import geopandas as gpd
import pandas as pd
import os

def get_geodataframe(config, longitude='longitude', latitude='latitude', crs='EPSG:4326'):
    """Georeferences a pandas dataframe using longitude and latitude columns of that dataframe. 

    Arguments:
        config {configuration}: parsed configuration settings

    Keyword Arguments:
        longitude {str}: column name with longitude coordinates (default: 'longitude')
        latitude {str}: column name with latitude coordinates (default: 'latitude')
        crs {str}: coordinate system to be used for georeferencing (default: 'EPSG:4326')

    Returns:
        gdf {geodataframe}: geodataframe containing entries with conflicts
    """    

    # construct path to file with conflict data
    conflict_fo = os.path.join(config.get('general', 'input_dir'), 
                               config.get('conflict', 'conflict_file'))

    # read file to pandas dataframe
    print('reading csv file to dataframe...' + os.linesep)
    df = pd.read_csv(conflict_fo)

    print('...translating to geopandas dataframe')
    
    gdf = gpd.GeoDataFrame(df,
                          geometry=gpd.points_from_xy(df[longitude], df[latitude]),
                          crs=crs)
    
    return gdf