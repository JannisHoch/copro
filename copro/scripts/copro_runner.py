from copro import settings, selection, evaluation, io, models, xydata

import click
import numpy as np
import pandas as pd
import os

import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("cfg", type=click.Path())
def cli(cfg):
    """Main command line script to execute the model.
    All settings are read from cfg-file.
    One cfg-file is required argument to train, test, and evaluate the model.
    Multiple classifiers are trained based on different train-test data combinations.
    Additional cfg-files for multiple projections can be provided as optional arguments,
    whereby each file corresponds to one projection to be made.
    Per projection, each classifiers is used to create separate projection outcomes per time step (year).
    All outcomes are combined after each time step to obtain the common projection outcome.

    Args:
        CFG (str): (relative) path to cfg-file
    """

    # - parsing settings-file
    # - returns dictionary with config-objects and output directories of reference run and all projections
    # - also returns root_dir which is the path to the cfg-file
    main_dict, root_dir = settings.initiate_setup(cfg)

    # - get config-objct and out_dir for reference run
    config_REF = main_dict["_REF"][0]
    out_dir_REF = main_dict["_REF"][1]

    click.echo(click.style("\nINFO: reference run started\n", fg="cyan"))

    # - selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_active_polys_gdf, global_df = selection.select(
        config_REF, out_dir_REF, root_dir
    )

    XY_class = xydata.XYData(config_REF)
    X, Y = XY_class.create_XY(
        out_dir=out_dir_REF,
        root_dir=root_dir,
        polygon_gdf=extent_active_polys_gdf,
        conflict_gdf=conflict_gdf,
    )

    # - create X and Y arrays by reading conflict and variable files for reference run
    # - or by loading a pre-computed array (npy-file) if specified in cfg-file
    # X, Y = xydata.create_XY(
    #     config_REF, out_dir_REF, root_dir, extent_active_polys_gdf, conflict_gdf
    # )

    # - defining scaling and model algorithms
    MachineLearning = models.MainModel(
        config=config_REF,
        X=X,
        Y=Y,
        out_dir=out_dir_REF,
    )

    # - fit-transform on scaler to be used later during projections
    click.echo("INFO: fitting scaler to sample data")
    # TODO: scaler_fitted needs to be part of the class
    _ = MachineLearning.scaler.fit(X[:, 2:])  # returns scaler_fitted

    # - initializing output variables
    out_X_df = pd.DataFrame()
    out_y_df = pd.DataFrame()
    out_dict = evaluation.init_out_dict()

    # - go through all n model executions
    # - that is, create different classifiers based on different train-test data combinations
    # TODO: this loop should be part of MachineLearning class
    click.echo("Training and testing machine learning model")
    for n in range(config_REF.getint("machine_learning", "n_runs")):

        click.echo(f"Run {n+1} of {config_REF.getint('machine_learning', 'n_runs')}.")

        # - run machine learning model and return outputs
        X_df, y_df, eval_dict = MachineLearning.run(run_nr=n)

        # - append per model execution
        out_X_df = pd.concat([out_X_df, X_df], axis=0, ignore_index=True)
        out_y_df = pd.concat([out_y_df, y_df], axis=0, ignore_index=True)
        out_dict = evaluation.fill_out_dict(out_dict, eval_dict)

    # - save output dictionary to csv-file
    io.save_to_csv(out_dict, out_dir_REF, "evaluation_metrics")
    io.save_to_npy(out_y_df, out_dir_REF, "raw_output_data")

    # - print mean values of all evaluation metrics
    for key, value in out_dict.items():
        click.echo(
            "Average {} of run with {} repetitions is {:0.3f}".format(
                key,
                config_REF.getint("machine_learning", "n_runs"),
                np.mean(value),
            )
        )

    # - create accuracy values per polygon and save to output folder
    gdf_hit = evaluation.polygon_model_accuracy(out_y_df, global_df)
    gdf_hit.to_file(
        os.path.join(out_dir_REF, "output_for_REF.geojson"), driver="GeoJSON"
    )

    click.echo(click.style("\nINFO: reference run succesfully finished\n", fg="cyan"))

    click.echo(click.style("INFO: starting projections\n", fg="cyan"))

    # - running prediction runs
    # TODO: scaler_fitted is now not part of the class
    MachineLearning.run_prediction(main_dict, root_dir, extent_active_polys_gdf)

    click.echo(click.style("\nINFO: all projections succesfully finished\n", fg="cyan"))
