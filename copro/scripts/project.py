#!/usr/bin/env python
"""
Run projections from a CoPro DEPLOY model.

WHY THIS EXISTS
---------------
Projection from the evaluation run is retired. `copro_runner`'s estimators are
fitted on the training portion only - 70% of rows under `random` and
`leave_polygons_out`, or years <= `temporal_cutoff` under `temporal` - so every
projection made that way came from a model that deliberately never saw part of
its own history.

Projections now come from the deploy model, fitted on all rows by
`copro/scripts/deploy.py`, with the scaler fitted on those same rows. This
script is the entry point that applies it.

It is separate from the deploy fit on purpose: a deploy model is fitted once
per region and predictor set, and projected from as often as wanted - one
scenario, or a hundred and fifty climate draws - without refitting anything.

WHAT IT READS AND WRITES
------------------------
    <deploy_dir>/_DEPLOY/       the fitted model and everything it came with
    <deploy_dir>/_PROJ/<key>/   one folder per scenario, written here

Everything the projection needs is inside `_DEPLOY`:

  - `copy_of_<config>.yaml`   the config the model was fitted from, including
                              its `projections` block, so scenario keys come
                              from the fit rather than from a config that may
                              have moved on
  - `copy_of_<key>.yaml`      the projection config per scenario key
  - `deploy_manifest.json`    feature columns in order, covered polygons
  - `estimator_deploy.pkl`, `scaler_deploy.pkl`
  - `selected_polygons.gpkg`  the polygons the fit selected
  - `files/conflicts_in_<y_end>.csv`, `files/last_conflict_year_<y_end>.csv`

`selection.select()` is deliberately NOT re-run. It would re-derive the polygon
set from the conflict file and the polygon file, and if either has changed
since the fit the projection would quietly run on a different set of polygons
than the model was fitted against. The stored layer cannot drift.

Like the deploy fit, this never calls `settings.initiate_setup()` and deletes
nothing. Re-running a scenario key overwrites that key's own output and leaves
every other key alone.

Usage:
    python -m copro.scripts.project <deploy_dir>
    python -m copro.scripts.project <deploy_dir> --keys proj_2026
"""

import argparse
import glob
import json
import os

import click
import geopandas as gpd

from copro import models, settings


def run_projection(deploy_dir, keys=None):
    """Runs the requested projections from a deploy model. Returns the output."""

    deploy_dir = os.path.abspath(deploy_dir)
    out_dir_DEPLOY = os.path.join(deploy_dir, "_DEPLOY")

    if not os.path.isdir(out_dir_DEPLOY):
        raise ValueError(
            f"No _DEPLOY folder in {deploy_dir} - fit the deploy model first."
        )

    # The config the model was fitted from. Read from _DEPLOY rather than from
    # CONFIG_DIR: the config there is regenerated on every run and may describe
    # a different predictor set than the one this model was fitted on.
    config_matches = sorted(glob.glob(os.path.join(out_dir_DEPLOY, "copy_of_*.yaml")))
    if len(config_matches) != 1:
        raise ValueError(
            f"Expected exactly one reference config copy in {out_dir_DEPLOY}, "
            f"found {len(config_matches)}: {[os.path.basename(f) for f in config_matches]}"
        )
    config_file = config_matches[0]
    config_DEPLOY = settings._parse_settings(config_file)

    # Paths inside the projection configs resolve against this. general.input_dir
    # is absolute, so the NetCDF reads are unaffected either way; what this does
    # is make the projection configs resolve next to the config that names them.
    root_dir = out_dir_DEPLOY

    manifest_file = os.path.join(out_dir_DEPLOY, "deploy_manifest.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "projections" not in config_DEPLOY:
        raise ValueError(
            f"No projections block in {os.path.basename(config_file)} - the "
            "deploy fit was made from a config with projections disabled."
        )

    available = list(config_DEPLOY["projections"].keys())
    if keys is None:
        keys = available
    else:
        unknown = [k for k in keys if k not in available]
        if unknown:
            raise ValueError(
                f"Unknown projection key(s) {unknown}. "
                f"This deploy model has: {available}"
            )

    click.echo(click.style("\nINFO: projection started\n", fg="cyan"))
    click.echo(f"Deploy model: {out_dir_DEPLOY}")
    click.echo(f"Feature set:  {manifest['feature_set']}")
    click.echo(f"Fitted:       {manifest['created_utc']}")
    click.echo(f"Scenarios:    {keys}")

    # run_prediction loops over this, so trimming it here is what restricts the
    # run to the requested keys.
    config_DEPLOY["projections"] = {
        k: config_DEPLOY["projections"][k] for k in keys
    }

    main_dict = {"_DEPLOY": [config_DEPLOY, out_dir_DEPLOY]}
    for key in keys:
        proj_file = os.path.join(root_dir, config_DEPLOY["projections"][key]["file"])
        config_PROJ = settings._parse_settings(proj_file)
        out_dir_PROJ = os.path.join(deploy_dir, "_PROJ", str(key))
        os.makedirs(out_dir_PROJ, exist_ok=True)
        # shape matches what run_prediction expects: main_dict[key][0][0] is the
        # config, main_dict[key][1] the out-dir
        main_dict[str(key)] = [[config_PROJ], out_dir_PROJ]

    polygons_file = os.path.join(out_dir_DEPLOY, "selected_polygons.gpkg")
    click.echo(f"Reading polygons from {polygons_file}.")
    selected_polygons_gdf = gpd.read_file(polygons_file)
    if "watprovID" not in selected_polygons_gdf.columns:
        raise ValueError(
            f"{polygons_file} has no watprovID column - "
            f"found {list(selected_polygons_gdf.columns)}."
        )

    all_y_df = models.run_prediction(main_dict, root_dir, selected_polygons_gdf)

    click.echo(click.style("\nINFO: projection finished\n", fg="cyan"))
    for key in keys:
        click.echo(f"Wrote {os.path.join(deploy_dir, '_PROJ', str(key))}")

    return all_y_df


def main():
    ap = argparse.ArgumentParser(
        description="Run projections from a fitted CoPro deploy model."
    )
    ap.add_argument(
        "deploy_dir",
        help="the deploy simulation folder holding _DEPLOY (and, after this "
        "run, _PROJ)",
    )
    ap.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="projection keys to run. Default: every key in the deploy "
        "model's own config.",
    )
    args = ap.parse_args()

    run_projection(deploy_dir=args.deploy_dir, keys=args.keys)


if __name__ == "__main__":
    main()
