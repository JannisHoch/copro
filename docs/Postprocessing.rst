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
        outputted geoJSON-files over time. The script uses all geoJSON-files
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

avg_over_time.py
-----------------

.. code-block:: console

    Usage: python avg_over_time.py [OPTIONS] INPUT_DIR OUTPUT_DIR SELECTED_POLYGONS

        Post-processing script to calculate average model output over a user-
        specifeid period or all output geoJSON-files stored in input-dir.
        Computed average values can be outputted as geoJSON-file or png-file or both.

        Args:     
            input_dir: path to input directory with geoJSON-files located per projection year.     
            output_dir (str): path to directory where output will be stored.     
            selected_polygons (str): path to a shp-file with all polygons used in a CoPro run.

        Output:     
            geoJSON-file with average column value per polygon (if geojson is set).     
            png-file with plot of average column value per polygon (if png is set)

        Options:
            -t0, --start-year INTEGER
            -t1, --end-year INTEGER
            -c, --column TEXT          column name
            --geojson / --no-geojson   save output to geojson or not
            --png / --no-png           save output to png or not
            --verbose / --no-verbose   verbose on/off

:command:`whaat is tisss` 

plot_polygon_vals.py
-----------------------

geojson2gif.py
---------------
