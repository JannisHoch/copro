#!/usr/bin/env python
"""
Fit the DEPLOY model for CoPro.
(CoPro pre-processing for projections - run once per region and predictor set,
before any projection.)

WHY THIS EXISTS
---------------
`copro_runner` fits its estimators inside `MainModel.run()`, which calls
`split_scale_train_test_split()` first. Those estimators are therefore trained
on the TRAINING PORTION ONLY - 70% of rows under `random`/`leave_polygons_out`,
or years <= `temporal_cutoff` under `temporal`. Projection from those estimators
is retired: projections come from the deploy model only.

There is a second, quieter problem this also removes. Two different scaler
instances used to be in play:

  - `MainModel.scaler` -> `scaler_all_data`, fitted on ALL rows in `__init__`,
    and passed to `predictive()` during projections.
  - `MachineLearning.scaler`, a fresh instance per `_n_run()`, fitted on the
    TRAIN rows only, which is the scaling each estimator actually learned under.

`QuantileTransformer` is rank-based, so a quantile map built from 70% of the
rows is not the map built from 100% of them. Here estimator and scaler are
fitted on the same rows and pickled together, so they cannot be mismatched.

WHAT THIS DOES
--------------
Fits ONE estimator on ALL rows, with the scaler fitted on those same rows:

    <out_dir>/_DEPLOY/estimator_deploy.pkl     RandomForest fitted on all rows
    <out_dir>/_DEPLOY/scaler_deploy.pkl        scaler fitted on the same rows
    <out_dir>/_DEPLOY/deploy_manifest.json     feature set, feature columns IN
                                               ORDER, covered polygons, row
                                               counts, hyperparameters, versions
    <out_dir>/_DEPLOY/copy_of_<config>.yaml    the config this fit came from
    <out_dir>/_DEPLOY/selected_*.gpkg          written by selection.select
    <out_dir>/_DEPLOY/files/conflicts_in_<y_end>.csv
    <out_dir>/_DEPLOY/files/last_conflict_year_<y_end>.csv
                                               projection initial conditions

Projections for this model are written to <out_dir>/_PROJ/<key>/ by the
projection run, so a deploy folder holds `_DEPLOY` and `_PROJ` as siblings.
The folder is deliberately NOT called `_REF`: in an evaluation run `_REF` holds
whatever the last split strategy left behind, and this model is fitted on
everything.

There is NO test set, NO evaluation metrics and NO split strategy here. Skill
is measured by the evaluation runs (`copro_runner` with `split_strategy` set);
this script only produces the model that gets deployed. `split_strategy` in the
config is ignored, deliberately - it is an evaluation-only setting.

HYPERPARAMETERS: THE SAME PROCEDURE THE EVALUATION RAN
------------------------------------------------------
The deploy model must be the model the evaluation runs measured. Nothing about
the hyperparameters is decided in this script, so it cannot drift from them:

  - `settings.define_model()` builds the estimator, exactly as it does for
    `copro_runner`: `settings.DEFAULT_HYPERPARAMETERS`, with any key in the
    config's optional `machine_learning.hyperparameters` block replacing the
    default for that key.
  - `settings.define_tuning()` says whether a search runs first
    (`machine_learning.tune_hyperparameters`, off when the config is silent).
  - `machine_learning.fit_estimator()` does the fit - the function the
    evaluation run calls on its training rows, called here on all rows.

Search off (the package default): the hyperparameters are applied as defined.

Search on: the same GridSearchCV the evaluation ran, on all rows, with the same
inner CV - `StratifiedGroupKFold` grouped on polygon ID, so whole polygons stay
inside one inner fold. The evaluation then measured "search, then fit", and
that is what is deployed.

Either way the manifest records what was fitted, whether it was searched, and
which hyperparameters came from the config rather than the package default.

ONE estimator, not n_runs of them: the spread across `n_runs` in the evaluation
comes from varying the SPLIT, and an all-data fit has no split to vary. A
200-tree forest is already an ensemble. If a spread over deploy fits is wanted
later, vary `random_state` and write estimator_deploy_<k>.pkl.

COVERED POLYGONS
----------------
`_split_XY_data` drops every row with a missing value, so some polygons have no
model at all (MENA: 4 of 187, on gaps in GDI, HDI and subnationalcorruption_SCI).
The manifest records the polygons that survived. Projections score those and no
others, so an unscored province is unscored by declaration rather than by an
accident of what the projection happened to find.

NOTE ON THE XY CACHE
--------------------
`create_XY` honours `pre_calc.XY` exactly as the reference run does, and
`_check_XY_cache` validates columns, y_start, y_end, polygon file and conflict
file. Point this at the same cache the evaluation run used and the feature
matrix is byte-identical with no NetCDF reads. The cache does NOT validate
NetCDF *contents*, so delete it after editing indicator rasters.

Reads : the reference config yaml (same one `copro_runner` takes)
Writes: <out_dir>/_DEPLOY as above

Usage:
    python -m copro.scripts.deploy <config.yaml> --out-dir <dir> --cores 8
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone

import click
import numpy as np
import sklearn

from copro import machine_learning, selection, settings, xydata


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hyperparameters(estimator):
    """The subset of RF attributes worth recording in the manifest.

    Same attribute list as copro_functions.R's hyperparameter extractor, so the
    deploy fit can be compared against the evaluation fits without a second
    convention. dicts (class_weight) are stringified the same way.
    """
    attrs = [
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "min_impurity_decrease",
        "max_features",
        "bootstrap",
        "class_weight",
        "random_state",
    ]
    out = {}
    for a in attrs:
        val = getattr(estimator, a, None)
        if isinstance(val, dict):
            val = ";".join(f"{k}:{v}" for k, v in val.items())
        out[a] = val
    try:
        out["n_trees_fitted"] = len(estimator.estimators_)
    except Exception:
        out["n_trees_fitted"] = None
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_deploy(
    config_file,
    out_dir,
    cores=8,
):
    """Fits the deploy model and writes the artifacts. Returns the manifest."""

    config_file = os.path.abspath(config_file)
    root_dir = os.path.dirname(config_file)

    # _parse_settings is nominally private but it is the canonical parser and
    # echoes what it read; duplicating yaml.safe_load here would be a second
    # convention for no gain.
    config = settings._parse_settings(config_file)

    # The predictor set is stated by the config, never inferred from the output
    # path. A folder can be labelled wrongly - the first MENA deploy fit landed
    # in a folder called `full` from a core config - and if the label were
    # derived from that path the artifacts would simply agree with the mistake.
    # Stated here, path and manifest can disagree, which is what makes it
    # visible.
    feature_set = config["general"]["feature_set"]

    # Deliberately NOT settings.initiate_setup(): that calls
    # io.make_and_collect_output_dirs, which shutil.rmtree()s the simulation
    # folder. A deploy fit must never delete anything.
    out_dir_DEPLOY = os.path.join(out_dir, "_DEPLOY")
    os.makedirs(out_dir_DEPLOY, exist_ok=True)
    shutil.copyfile(
        config_file,
        os.path.join(out_dir_DEPLOY, f"copy_of_{os.path.basename(config_file)}"),
    )

    # The projection configs travel with the model too, under their own names,
    # so the bare filename in the `projections` block resolves inside _DEPLOY
    # and the projection run needs nothing from CONFIG_DIR - where configs are
    # regenerated on every run and may describe a different predictor set.
    for key, entry in config.get("projections", {}).items():
        proj_src = os.path.join(root_dir, entry["file"])
        shutil.copyfile(
            proj_src, os.path.join(out_dir_DEPLOY, os.path.basename(proj_src))
        )
        click.echo(f"Copied projection config for '{key}': {entry['file']}")

    click.echo(click.style("\nINFO: deploy fit started\n", fg="cyan"))
    click.echo(f"Config:      {config_file}")
    click.echo(f"Feature set: {feature_set}")
    click.echo(f"Output dir:  {out_dir_DEPLOY}")

    estimator = settings.define_model(config)
    tune_hyperparameters = settings.define_tuning(config)
    target_var = settings.define_target_var(config, estimator)

    # same selection step as the reference run; writes selected_conflicts.gpkg
    # and selected_polygons.gpkg into out_dir_DEPLOY
    conflict_gdf, extent_active_polys_gdf, global_df = selection.select(
        config, out_dir_DEPLOY, root_dir
    )

    # same feature matrix as the reference run. This also writes
    # files/conflicts_in_<y_end>.csv and files/last_conflict_year_<y_end>.csv,
    # which are the projection initial conditions.
    # write_xy_gpkg=False: XY.gpkg is ~692 MB per region and nothing reads it.
    XY_class = xydata.XYData(config, target_var)
    X, Y, meta = XY_class.create_XY(
        out_dir=out_dir_DEPLOY,
        root_dir=root_dir,
        polygon_gdf=extent_active_polys_gdf,
        conflict_gdf=conflict_gdf,
        write_xy_gpkg=False,
    )

    # Columns 0 and 1 are poly_ID and poly_geometry (see
    # machine_learning._split_conflict_geom_data); everything after is a
    # feature, in the order the estimator will expect them forever after.
    feature_cols = [str(c) for c in X.columns[2:]]
    X_data = X.iloc[:, 2:].to_numpy()
    Y_arr = np.asarray(Y).astype(int)

    # Polygons that survived the dropna in _split_XY_data. Projections score
    # these and no others.
    covered_poly_IDs = sorted(set(X["poly_ID"].tolist()))

    n_rows = int(X_data.shape[0])
    n_pos = int(Y_arr.sum())
    click.echo(
        f"\nFitting on ALL {n_rows} rows "
        f"({n_pos} positive, {100 * n_pos / n_rows:.2f}%), "
        f"{len(feature_cols)} features, "
        f"{len(covered_poly_IDs)} polygons covered."
    )
    if n_pos == 0:
        raise ValueError("No positive cases in the target - refusing to fit.")

    # --- scaler: fitted on the same rows the estimator is fitted on ---------
    scaler = machine_learning.define_scaling(config)
    X_scaled = scaler.fit_transform(X_data)

    # --- estimator: the same fit the evaluation run does --------------------
    # Hyperparameters were set by settings.define_model above. The polygon IDs
    # and the seed are used only if the search is on: same grouping and same
    # seed convention as run 0 of the evaluation.
    hyperparameters_from_config = sorted(
        config["machine_learning"].get("hyperparameters", {}).keys()
    )
    tuning_seed = config["machine_learning"].get("split_seed", 42)
    fitted_estimator = machine_learning.fit_estimator(
        estimator,
        X_scaled,
        Y_arr,
        X["poly_ID"].to_numpy(),
        tune_hyperparameters,
        tuning_seed,
        n_jobs=cores,
    )

    # --- write --------------------------------------------------------------
    est_path = os.path.join(out_dir_DEPLOY, "estimator_deploy.pkl")
    scl_path = os.path.join(out_dir_DEPLOY, "scaler_deploy.pkl")
    with open(est_path, "wb") as f:
        pickle.dump(fitted_estimator, f)
    with open(scl_path, "wb") as f:
        pickle.dump(scaler, f)

    manifest = {
        "kind": "copro_deploy_model",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulation_name": config["general"].get("simulation_name"),
        "feature_set": feature_set,
        "config_file": os.path.basename(config_file),
        "y_start": config["general"]["y_start"],
        "y_end": config["general"]["y_end"],
        "scaler": config["machine_learning"]["scaler"],
        # order matters: the projection must present columns in exactly this
        # order, and nothing downstream can check it by name (see
        # machine_learning.check_projection_columns)
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "covered_poly_IDs": covered_poly_IDs,
        "n_covered_polygons": len(covered_poly_IDs),
        "n_rows": n_rows,
        "n_positive": n_pos,
        "positive_fraction": round(n_pos / n_rows, 6),
        # the values as fitted - after the search, if there was one
        "hyperparameters": _hyperparameters(fitted_estimator),
        # "default": settings.DEFAULT_HYPERPARAMETERS untouched.
        # "config": at least one key came from machine_learning.hyperparameters.
        "hyperparameters_source": "config" if hyperparameters_from_config else "default",
        "hyperparameters_from_config": hyperparameters_from_config,
        # True: n_estimators, min_samples_leaf and class_weight above were
        # selected by GridSearchCV on all rows and override default and config.
        "tuned": tune_hyperparameters,
        "tuning_seed": tuning_seed if tune_hyperparameters else None,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "estimator_file": os.path.basename(est_path),
        "scaler_file": os.path.basename(scl_path),
        # stated explicitly so nothing downstream infers a split from the config
        "split_strategy": None,
        "note": (
            "Fitted on all rows; no test set and no metrics. Skill comes from "
            "the evaluation runs, not from this fit."
        ),
    }
    man_path = os.path.join(out_dir_DEPLOY, "deploy_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    click.echo(f"\nWrote {est_path}")
    click.echo(f"Wrote {scl_path}")
    click.echo(f"Wrote {man_path}")

    files_dir = os.path.join(out_dir_DEPLOY, "files")
    y_end = config["general"]["y_end"]
    for fname in (
        f"conflicts_in_{y_end}.csv",
        f"last_conflict_year_{y_end}.csv",
    ):
        fpath = os.path.join(files_dir, fname)
        if os.path.isfile(fpath):
            click.echo(f"Projection initial conditions present: {fpath}")
        else:
            # loud, not fatal: the projection cannot anchor conflict at t-1
            # without these, so better to know now than at projection time.
            click.echo(
                click.style(
                    f"WARNING: expected {fpath} but it is missing - "
                    "the projection will have no observed conflict anchor.",
                    fg="red",
                )
            )

    click.echo(click.style("\nINFO: deploy fit finished\n", fg="cyan"))
    return manifest


def main():
    ap = argparse.ArgumentParser(
        description="Fit the CoPro deploy model on all available rows."
    )
    ap.add_argument("config", help="path to the reference config yaml")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="simulation folder for this deploy model; artifacts are written "
        "to <out-dir>/_DEPLOY and projections later to <out-dir>/_PROJ "
        "(created if absent; nothing is deleted)",
    )
    ap.add_argument("--cores", type=int, default=8, help="n_jobs. Default 8.")
    args = ap.parse_args()

    build_deploy(
        config_file=args.config,
        out_dir=args.out_dir,
        cores=args.cores,
    )


if __name__ == "__main__":
    main()
