import conflict_model 

from configparser import RawConfigParser

import click
from os.path import isdir, dirname, abspath
from os import makedirs
import geopandas as gpd
import os, sys

# ad-hoc functions
def parse_dir(param, path):
    try:
        path = abspath(path)
        if not isdir(path):
            os.makedirs(path)
    except:
        raise click.BadParameter("Couldn't understand or create folder directory for the '{}' argument.".format(param))
    return path

@click.group()
def cli():
    pass

@click.command()
@click.argument('cfg',)
@click.option('-o', '--out-dir', default=None, help='directory to save model outputs', type=click.Path(), show_default=True)
@click.option('-s', '--safe-plots', default=False, help='whether or not to safe plots', type=bool, show_default=True)

def main(cfg, out_dir=None, safe_plots=False):
    """
    Runs the conflict_model from command line with several options and the settings cfg-file as argument.

    CFG: path to cfg-file with run settings
    """

    if gpd.__version__ < '0.7.0':
        sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}'.format(gpd.__version__))

    if out_dir:
        out_dir = parse_dir('out-dir', out_dir)

    config = RawConfigParser(allow_no_value=True)
    config.read(cfg)

    conflict_gdf = conflict_model.utils.get_geodataframe(config)

    selected_conflict_gdf, continent_gdf = conflict_model.selection.select(conflict_gdf, config)

    conflict_model.analysis.conflict_in_year_bool(selected_conflict_gdf, continent_gdf, config, saving_plots=safe_plots, out_dir=out_dir)

if __name__ == '__main__':
    main()