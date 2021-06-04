import click
import glob
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import os

@click.command()
@click.option('-id', '--polygon-id', multiple=True, type=str)
@click.option('-c', '--column', help='column name', default='chance_of_conflict', type=str)
@click.option('-t', '--title', help='title for plot and file_object name', type=str)
@click.option('--verbose/--no-verbose', help='verbose on/off', default=False)
@click.argument('input-dir', type=click.Path())
@click.argument('output-dir', type=click.Path())

def main(input_dir=None, polygon_id=None, column=None, title=None, output_dir=None, verbose=None):
    """Quick and dirty function to plot the develoment of a column in the outputted geojson-files over time.
    """

    click.echo('\nPLOTTING VARIABLE DEVELOPMENT OVER TIME')

    # converting polygon IDs to list
    polygon_id = list(polygon_id)

    # check that there is at least one ID or 'all' specified
    assert(len(polygon_id) > 0), AssertionError('ERROR: please specify at least one polygon ID to be sampled or select ''all'' for sampling the entire study area')

    # if 'all' is specified, no need to have list but get value directly
    if polygon_id[0] == 'all':
        click.echo('INFO: selected entire study area')
        polygon_id = 'all'

    # absolute path to input_dir
    input_dir = os.path.abspath(input_dir)
    click.echo('INFO: getting geojson-files from {}'.format(input_dir))

    # collect all files in input_dir
    all_files = glob.glob(os.path.join(input_dir, '*.geojson'))
    
    if verbose: 
        if polygon_id != 'all': 
            click.echo('DEBUG: sampling from IDs'.format(polygon_id))

    # create dictionary with list for areas (either IDs or entire study area) to be sampled from
    out_dict = dict()
    if polygon_id != 'all':
        for idx in polygon_id:
                out_dict[int(idx)] = list()
    else:
        out_dict[polygon_id] = list()

    # create a list to keep track of year-values in files
    years = list()
    
    # go through all files
    click.echo('INFO: retrieving values from column {}'.format(column))
    for geojson in all_files:

        if verbose: click.echo('DEBUG: reading file {}'.format(geojson))
        # read file and convert to geo-dataframe
        gdf = gpd.read_file(geojson, driver='GeoJSON')
        # convert geo-dataframe to dataframe
        df = pd.DataFrame(gdf.drop(columns='geometry'))

        # get year-value
        year = int(str(str(os.path.basename(geojson)).rsplit('.')[0]).rsplit('_')[-1])
        years.append(year)

        # go throough all IDs
        if polygon_id != 'all':
            for idx in polygon_id:
                if verbose: 
                    click.echo('DEBUG: sampling ID {}'.format(idx))

                # if ID not in file, assign NaN
                if int(idx) not in df.ID.values: 
                    click.echo('WARNING: ID {} is not in {} - NaN set'.format(int(idx), geojson))
                    vals = np.nan
                # otherwise, get value of column at this ID
                else:
                    vals = df[column].loc[df.ID==int(idx)].values[0]

                # append this value to list in dict
                idx_list = out_dict[int(idx)]
                idx_list.append(vals)

        else:
            # compute mean value over column
            vals = df[column].mean()
            # append this value to list in dict
            idx_list = out_dict[polygon_id]
            idx_list.append(vals)


    # create a dataframe from dict and assign year-values as index
    df = pd.DataFrame().from_dict(out_dict)
    years = pd.to_datetime(years, format='%Y')
    df.index = years

    # create an output folder, if not yet there
    if not os.path.isdir(os.path.abspath(output_dir)):
        click.echo('INFO: creating output folder {}'.format(os.path.abspath(output_dir)))
        os.makedirs(os.path.abspath(output_dir))

    # save dataframe as csv-file
    if polygon_id != 'all':
        click.echo('INFO: saving to file {}'.format(os.path.abspath(os.path.join(output_dir, '{}_dev_IDs.csv'.format(column)))))
        df.to_csv(os.path.abspath(os.path.join(output_dir, '{}_dev_IDs.csv'.format(column))))
    else:
        click.echo('INFO: saving to file {}'.format(os.path.abspath(os.path.join(output_dir, '{}_dev_all.csv'.format(column)))))
        df.to_csv(os.path.abspath(os.path.join(output_dir, '{}_dev_all.csv'.format(column))))

    # create a simple plot and save to file
    # if IDs are specified, with one subplot per ID
    if polygon_id != 'all':
        fig, axes = plt.subplots(nrows=len(polygon_id), ncols=1, sharex=True)
        df.plot(subplots=True, ax=axes)
        for ax in axes:
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.1, 1))
        if title != None:
            ax.set_title(str(title))
        plt.savefig(os.path.abspath(os.path.join(output_dir, '{}_dev_IDs.png'.format(column))), dpi=300, bbox_inches='tight')
    # otherwise, only one plot needed
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        df.plot(ax=ax)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 1))
        if title != None:
            ax.set_title(str(title))
        plt.savefig(os.path.abspath(os.path.join(output_dir, '{}_dev_all.png'.format(column))), dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    main()