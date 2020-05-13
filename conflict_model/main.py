import selection, utils
import geopandas as gpd
from configparser import RawConfigParser

if gpd.__version__ < '0.7.0':
    sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}'.format(gpd.__version__))

settings_file = r'../data/run_setting.cfg'

config = RawConfigParser(allow_no_value=True)
config.read(settings_file)

conflict_gdf = utils.get_geodataframe(config)

print(conflict_gdf.head())

selected_conflict_gdf, continent_gdf = selection.select(conflict_gdf, config)