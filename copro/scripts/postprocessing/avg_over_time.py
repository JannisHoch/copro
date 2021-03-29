import click
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from copro import utils, evaluation
import os, sys

@click.command()
@click.option('-t0', '--start-year', type=int)
@click.option('-t1', '--end-year', type=int)
@click.option('-c', '--column', help='column name', default='chance_of_conflict', type=str)
@click.option('--geojson/--no-geojson', help='save output to geojson or not', default=False)
@click.option('--png/--no-png', help='save output to png or not', default=True)
@click.option('--verbose/--no-verbose', help='verbose on/off', default=False)
@click.argument('input_dir', type=click.Path())
@click.argument('output_dir',type=click.Path())
@click.argument('selected_polygons', type=click.Path())

def main(input_dir=None, output_dir=None, selected_polygons=None, start_year=None, end_year=None, geojson=None, png=None, column=None, verbose=None):
    """Post-processing script to calculate average model output over a user-specifeid period or all output geojson-files.
    """    

    # check if start/end time settings are consistent
    if ((start_year != None) and (end_year == None)) or ((end_year != None) and (start_year == None)):
        raise ValueError('ERROR: if start or end year is specified, the pendant must be specified too!')

    # read a shp-file with geometries of all selected polygons
    click.echo('\nreading shp-file with all polygons from {}'.format(os.path.abspath(selected_polygons)))
    selected_polygons_gdf = gpd.read_file(os.path.abspath(selected_polygons))
    # create dataframe
    global_df = utils.global_ID_geom_info(selected_polygons_gdf)

    # find all geojson-files in input-dir
    input_dir = os.path.abspath(input_dir)
    click.echo('getting geojson-files from {}'.format(input_dir))
    all_files = sorted(glob.glob(os.path.join(input_dir, '*.geojson')))

    # if specific start/end time is specified, find only those geojson-files for specified period
    if (start_year != None) and (end_year != None):
        # define period between start and ent time
        period = np.arange(start_year, end_year+1, 1)
        click.echo('using all geojson-files for years {} to {}'.format(period[0], period[-1]))
        # creating suffix for file saving later
        suffix = '{}_to_{}'.format(period[0], period[-1])
        # initiate empty list for selected geojson-files
        selected_files = []
        # select
        for fo in all_files:
            # if the year-suffix of geojson-file matches year in period, add to list
            year = int(str(str(os.path.basename(fo)).rsplit('.')[0]).rsplit('_')[-1])
            if year in period:
                if verbose: print('adding to selection')
                selected_files.append(fo)
    # if not end/start time is specified, use all geojson-files in input-dir
    else:
        click.echo('using all geojson-files in input-dir')
        # also here, create suffix for file saving laster
        suffix = 'all_years'
        selected_files = all_files

    # initiatie empyt dataframe for collecting all annual output
    y_df = pd.DataFrame()

    # go through all geojson-files left over after selection step
    for geojson in selected_files:
        # read files and convert to datatrame
        if verbose: click.echo('reading file {}'.format(geojson))
        gdf = gpd.read_file(geojson, driver='GeoJSON')
        df = pd.DataFrame(gdf)
        # append to dataframe 
        y_df = y_df.append(df, ignore_index=True)
    
    # initiate dataframe for time-averaged output
    click.echo('creating one output dataframe from all geojson-files')
    y_out = pd.DataFrame()
    # get all unique IDs of polygons
    y_out['ID'] = y_df.ID.unique()
    click.echo('reading from column {}'.format(column))
    if column == 'chance_of_conflict':
        # add number of predictiosn made over all selected years
        y_out = pd.merge(y_out, y_df.nr_predictions.groupby(y_df.ID).sum().to_frame(), on='ID')
        # add number of predicted conflicts over all selected years
        y_out = pd.merge(y_out, y_df.nr_predicted_conflicts.groupby(y_df.ID).sum().to_frame(), on='ID')
        # determine chance of conflict over all selected years
        y_out[column] = y_out.nr_predicted_conflicts / y_out.nr_predictions
    elif column == 'avg_prob_1':
        y_out = pd.merge(y_out, pd.to_numeric(y_df[column]).groupby(y_df.ID).mean().to_frame(), on='ID')
    else:
        raise ValueError('ERROR: column {} is not yet supported'.format(column))
    # add geometry informatoin for each polygon
    y_out = pd.merge(y_out, global_df, on='ID', how='left')

    if not os.path.isdir(os.path.abspath(output_dir)):
        click.echo('creating output folder {}'.format(os.path.abspath(output_dir)))
        os.makedirs(os.path.abspath(output_dir))

    # convert to geo-dataframe
    gdf_out = gpd.GeoDataFrame(y_out, geometry=y_out.geometry)

    # if specified, save as geojson-file to output-dir
    if geojson:
        click.echo('saving to {}'.format(os.path.abspath(os.path.join(output_dir, '{}_merged_{}.geojson'.format(column, suffix)))))
        gdf_out.to_file(os.path.abspath(os.path.join(output_dir, '{}_merged_{}.geojson'.format(column, suffix))), driver='GeoJSON')

    # if specified, save as png-file to output-dir
    if png:
        fig, ax = plt.subplots(1, 1)
        gdf_out.plot(column=column, ax=ax, 
                        cmap='Reds', 
                        vmin=0, vmax=1,
                        legend=True,
                        legend_kwds={'label': column, 'orientation': "vertical"})

        click.echo('saving to {}'.format(os.path.abspath(os.path.join(output_dir, '{}_merged_{}.png'.format(column, suffix)))))
        plt.savefig(os.path.abspath(os.path.join(output_dir, '{}_merged_{}.png'.format(column, suffix))), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':

    main()