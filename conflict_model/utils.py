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
    print('reading csv file to dataframe {}'.format(conflict_fo))
    df = pd.read_csv(conflict_fo)
    print('...DONE' + os.linesep)

    print('translating to geopandas dataframe')
    gdf = gpd.GeoDataFrame(df,
                          geometry=gpd.points_from_xy(df[longitude], df[latitude]),
                          crs=crs)
    print('...DONE' + os.linesep)
    
    return gdf

def show_versions():
    from conflict_model import __version__ as cm_version
    from geopandas import __version__ as gpd_version
    from pandas import __version__ as pd_version
    from numpy import __version__ as np_version
    from matplotlib import __version__ as mpl_version
    from rasterstats import __version__ as rstats_version
    from xarray import __version__ as xr_version
    from rasterio import __version__ as rio_version
    from sys import version as os_version

    print("Python version: {}".format(os_version))
    print("conflict_model version: {}".format(cm_version))
    print("geopandas version: {}".format(gpd_version))
    print("xarray version: {}".format(xr_version))
    print("rasterio version: {}".format(rio_version))
    print("pandas version: {}".format(pd_version))
    print("numpy version: {}".format(np_version))
    print("matplotlib version: {}".format(mpl_version))
    print("rasterstats version: {}".format(rstats_version))