Postprocessing
=========================

There are several command line scripts available for post-processing. 
In addition to quick plots to evaluate model output, they also produce files for use in bespoke plotting and analysis scripts.

The scripts are located under ``/copro/scripts/postprocessing``.

plot_value_over_time.py
------------------------

.. code-block:: console

    Usage: python plot_value_over_time.py [OPTIONS] INPUT_DIR OUTPUT_DIR

        Quick and dirty function to plot the develoment of a column in the
        outputted geojson-files over time. The script uses all geoJSON-files
        located in input-dir and retrieves values from them. Possible to plot
        obtain development for multiple polygons (indicated via their ID) or
        entire study area. If the latter, then different statistics can be chosen
        (mean, max, min, std).

        Args:     
            input-dir (str): path to input directory with geoJSON-files located per projection year. 
            output-dir (str): path to directory where output will be stored.

        Output:     
            a csv-file containing values per time step.     
            a png-file showing development over time.

        Options:
        -id, --polygon-id TEXT
        -s, --statistics TEXT     which statistical method to use (mean, max, min,
                                    std). note: has only effect if with "-id all"!

        -c, --column TEXT         column name
        -t, --title TEXT          title for plot and file_object name
        --verbose / --no-verbose  verbose on/off
        --help                    Show this message and exit.
