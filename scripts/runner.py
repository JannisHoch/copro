import conflict_model 

from configparser import RawConfigParser

import click
from os.path import isdir, dirname, abspath
from os import makedirs
import geopandas as gpd
import numpy as np
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

def main(cfg, out_dir=None, safe_plots=False):
    """
    Runs the conflict_model from command line with several options and the settings cfg-file as argument.

    CFG: path to cfg-file with run settings
    """
    print('')
    print('#### LETS GET STARTED PEOPLZ! ####' + os.linesep)

    if gpd.__version__ < '0.7.0':
        sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}'.format(gpd.__version__))

    config = RawConfigParser(allow_no_value=True)
    config.read(cfg)

    #out_dir
    out_dir = config.get('general','output_dir')
    if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    print('for the record, saving output to folder {}'.format(out_dir) + os.linesep)

    conflict_gdf = conflict_model.utils.get_geodataframe(config)

    selected_conflict_gdf, extent_gdf = conflict_model.selection.select(conflict_gdf, config)

    sim_years = np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end'), 1)

    print('preps are all done, now entering annual analysis' + os.linesep)

    for sim_year in np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end'), 1):

        print('entering year {}'.format(sim_year) + os.linesep)

        conflict_gdf_perYear, extent_conflict_merged, fatalities_per_waterProvince, extent_waterProvinces_with_boolFatalities = conflict_model.analysis.conflict_in_year_bool(selected_conflict_gdf, extent_gdf, config, sim_year, out_dir, saving_plots=True)

        GDP_PPP_gdf = conflict_model.env_vars_nc.rasterstats_GDP_PPP(extent_waterProvinces_with_boolFatalities, extent_gdf, config, sim_year, out_dir, saving_plots=True)

if __name__ == '__main__':
    main()