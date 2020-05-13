import geopandas as gpd
import pandas as pd
import os

def get_geodataframe(config, longitude='longitude', latitude='latitude', crs='EPSG:4326'):

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