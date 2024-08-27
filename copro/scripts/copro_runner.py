from copro import settings, selection, evaluation, io, models, xydata

import click
import numpy as np
import pandas as pd
import os


@click.command()
@click.argument("cfg", type=click.Path())
@click.option(
    "--cores",
    "-c",
    type=int,
    default=5,
    help="Number of jobs to run in parallel. Default is 0.",
)
@click.option(
    "--verbose",
    "-v",
    type=int,
    default=0,
    help="Verbosity level of the output. Default is 0.",
)
def cli(cfg: click.Path, cores: int, verbose: int):
    """Main command line script to execute the model.

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

    estimator = settings.define_model(config_REF)
    target_var = settings.define_target_var(config_REF, estimator)

    # - selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_active_polys_gdf, global_df = selection.select(
        config_REF, out_dir_REF, root_dir
    )

    XY_class = xydata.XYData(config_REF, target_var)
    X, Y = XY_class.create_XY(
        out_dir=out_dir_REF,
        root_dir=root_dir,
        polygon_gdf=extent_active_polys_gdf,
        conflict_gdf=conflict_gdf,
    )

    # - defining scaling and model algorithms
    ModelWorkflow = models.MainModel(
        config=config_REF,
        X=X,
        Y=Y,
        estimator=estimator,
        out_dir=out_dir_REF,
        n_jobs=cores,
        verbose=verbose,
    )

    # - fit-transform on scaler to be used later during projections
    _, out_y_df, out_perm_importances_arr, out_dict = ModelWorkflow.run(
        config_REF["machine_learning"]["n_runs"], tune_hyperparameters=True
    )

    # - save output to files
    out_perm_importances_df = pd.DataFrame(
        data=out_perm_importances_arr,
        columns=[
            key
            for key in XY_class.XY_dict
            if key not in ["poly_ID", "poly_geometry", "conflict"]
        ],
    )
    out_perm_importances_df.to_parquet(
        os.path.join(out_dir_REF, "perm_importances.parquet")
    )
    io.save_to_csv(out_dict, out_dir_REF, "evaluation_metrics")
    io.save_to_npy(out_y_df, out_dir_REF, "raw_output_data")

    # - print mean values of all evaluation metrics
    for key, value in out_dict.items():
        click.echo(
            "Average {} of run with {} repetitions is {:0.3f}".format(
                key,
                config_REF["machine_learning"]["n_runs"],
                np.mean(value),
            )
        )

    # - create accuracy values per polygon and save to output folder
    gdf_hit = evaluation.polygon_model_accuracy(out_y_df, global_df)
    gdf_hit.to_file(
        os.path.join(out_dir_REF, "output_for_REF.geojson"), driver="GeoJSON"
    )

    click.echo(click.style("\nINFO: reference run succesfully finished\n", fg="cyan"))

    if "projections" in config_REF.keys():
        click.echo(click.style("INFO: starting projections\n", fg="cyan"))
        # - running prediction runs
        # TODO: scaler_fitted is now not part of the class
        ModelWorkflow.run_prediction(main_dict, root_dir, extent_active_polys_gdf)

    click.echo(click.style("\nINFO: all projections succesfully finished\n", fg="cyan"))
