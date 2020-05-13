import conflict_model 

from configparser import RawConfigParser

import click
import geopandas as gpd
import os, sys

@click.group()
def cli():
    pass

@click.command()
@click.argument('cfg',)
# @click.option('--env', default='', help='path to glofrim env file with engine paths')
# @click.option('-o', '--out-dir', default='', help='directory to save model outputs', type=click.Path())
# @click.option('-s', '--start-date', default='', help='set start time for all models')
# @click.option('-e', '--end-date', default='', help='set end time for all models')

def main(cfg):
    """
    CFG: path to cfg-file with run settings
    """
# def run(cfg, env='', out_dir='', end_date='', start_date=''):

    if gpd.__version__ < '0.7.0':
        sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}'.format(gpd.__version__))

    config = RawConfigParser(allow_no_value=True)
    config.read(cfg)

    conflict_gdf = conflict_model.utils.get_geodataframe(config)

    selected_conflict_gdf, continent_gdf = conflict_model.selection.select(conflict_gdf, config)

if __name__ == '__main__':
    main()