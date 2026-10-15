import json
import os
from datetime import datetime, timezone

import click
import numpy as np
import pandas as pd

from copro import evaluation, io, models, selection, settings, xydata


@click.command()
@click.argument("cfg", type=click.Path())
@click.option(
    "--cores",
    "-c",
    type=int,
    default=5,
    help="Number of jobs to run in parallel. Default is 5.",
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
    tune_hyperparameters = settings.define_tuning(config_REF)
    target_var = settings.define_target_var(config_REF, estimator)

    # The temporal split is a rule, not a draw: every run holds out the same
    # years. With the search off the forest seed, the scaler seed and the
    # permutation seed are fixed as well, so n_runs would be n_runs identical
    # fits - and identical copies read downstream as perfect run-to-run
    # agreement and zero spread. It is one measurement, so it is fitted once.
    # With the search on the runs do differ (the inner CV folds are seeded per
    # run and can select different hyperparameters), so n_runs is kept.
    n_runs = config_REF["machine_learning"]["n_runs"]
    split_strategy = config_REF["machine_learning"].get("split_strategy", "random")
    if split_strategy == "temporal" and not tune_hyperparameters:
        click.echo(
            f"Temporal split without hyperparameter search: fitting once, "
            f"not n_runs = {n_runs} times - the runs would be identical."
        )
        n_runs = 1

    # - selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_active_polys_gdf, global_df = selection.select(
        config_REF, out_dir_REF, root_dir
    )

    XY_class = xydata.XYData(config_REF, target_var)
    X, Y, meta = XY_class.create_XY(
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
        meta=meta,
        estimator=estimator,
        out_dir=out_dir_REF,
        n_jobs=cores,
        verbose=verbose,
    )

    # - fit-transform on scaler to be used later during projections
    _, out_y_df, out_perm_importances_arr, out_dict = ModelWorkflow.run(
        n_runs, tune_hyperparameters=tune_hyperparameters
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

    # inspection.permutation_importance() returns one row PER REPEAT
    # (n_repeats=10 in machine_learning.fit_predict), and MainModel.run() vstacks
    # one such block per run - so this frame has n_runs * n_repeats rows, not
    # n_runs. Unlabelled, a repeat is indistinguishable from a run downstream,
    # and correlating repeats of a single fit reads as rank stability across
    # fits, which makes importance look far more stable than it is. Label them
    # so the two can be told apart.
    n_runs_cfg = n_runs
    n_imp_rows = len(out_perm_importances_df)
    if n_imp_rows % n_runs_cfg != 0:
        raise ValueError(
            f"Permutation importance has {n_imp_rows} rows, which is not "
            f"divisible by n_runs ({n_runs_cfg}) - cannot label runs and "
            "repeats."
        )
    n_repeats = n_imp_rows // n_runs_cfg
    out_perm_importances_df.insert(
        0, "run", np.repeat(np.arange(n_runs_cfg), n_repeats)
    )
    out_perm_importances_df.insert(
        1, "repeat_nr", np.tile(np.arange(n_repeats), n_runs_cfg)
    )
    click.echo(
        f"Permutation importance: {n_runs_cfg} runs x {n_repeats} repeats "
        f"= {n_imp_rows} rows."
    )

    out_perm_importances_df.to_parquet(
        os.path.join(out_dir_REF, "perm_importances.parquet")
    )
    io.save_to_csv(out_dict, out_dir_REF, "evaluation_metrics")
    io.save_to_npy(out_y_df, out_dir_REF, "raw_output_data")

    # tidy per-test-row table for downstream evaluation.
    # Written to a persistent eval/ folder (sibling of the per-run _REF folder,
    # which is recreated each run) so multiple split strategies accumulate side
    # by side: <output_dir>/eval/<simulation_name>_<strategy>.csv
    row_table = out_y_df[
        ["run_nr", "ID", "year", "conflict_t_min_1", "y_test", "y_pred", "y_prob_1"]
    ].copy()
    for col in ["run_nr", "year", "conflict_t_min_1", "y_test", "y_pred"]:
        row_table[col] = pd.to_numeric(row_table[col]).astype(int)
    row_table["y_prob_1"] = pd.to_numeric(row_table["y_prob_1"])

    sim_name = config_REF["general"]["simulation_name"]
    eval_dir = os.path.join(root_dir, config_REF["general"]["output_dir"], "eval")
    os.makedirs(eval_dir, exist_ok=True)
    row_table.to_csv(
        os.path.join(eval_dir, f"{sim_name}_{split_strategy}.csv"), index=False
    )

    # What produced these rows, written next to them. The csv records nothing
    # about its run and eval/ is not cleared between runs, so a file left by an
    # earlier predictor set or hyperparameter configuration is indistinguishable
    # from a current one. scripts/calibrate.py compares `fit` with the deploy
    # model's configuration and refuses rows that do not match.
    rows_description = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulation_name": sim_name,
        "split_strategy": split_strategy,
        "n_runs": n_runs,
        "split_seed": config_REF["machine_learning"].get("split_seed", 42),
        "fit": settings.describe_fit(
            config_REF, estimator, tune_hyperparameters, X.columns[2:]
        ),
    }
    with open(
        os.path.join(eval_dir, f"{sim_name}_{split_strategy}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(rows_description, f, indent=2)

    # - print mean values of all evaluation metrics
    for key, value in out_dict.items():
        click.echo(
            "Average {} of run with {} repetitions is {:0.3f}".format(
                key,
                n_runs,
                np.mean(value),
            )
        )

    # - create accuracy values per polygon and save to output folder
    gdf_hit = evaluation.polygon_model_accuracy(out_y_df, global_df)
    gdf_hit.to_file(
        os.path.join(out_dir_REF, "output_for_REF.gpkg"), driver="GPKG", crs="EPSG:4326"
    )

    click.echo(click.style("\nINFO: reference run succesfully finished\n", fg="cyan"))