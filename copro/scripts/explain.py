#!/usr/bin/env python
"""
Attribute a CoPro deploy projection to its features with SHAP.
(CoPro postprocessing - run after copro.scripts.project.)

WHAT IT DOES
------------
For one projection key and year, splits every scored polygon's score into one
contribution per feature with TreeSHAP (interventional):

    base_value + sum(shap over features) = y_prob_1

base_value is the mean score over the BACKGROUND: a fixed-seed sample of the
region's EMERGING province-years (conflict_t_min_1 == 0), taken from the rows
the deploy model was fitted on. A contribution therefore says how much a
feature lifts a polygon above a typical peaceful polygon in its own region.

SHAP retains at most 100 background rows, so n_background above 100 is sampled 
and discarded; the baseline is the mean over the retained rows and carries their 
sampling spread, about 0.004 between draws in LAC.

Contributions are computed on the scaled features the model sees and reported
beside the raw value. QuantileTransformer is monotone per feature, so every
contribution belongs to exactly one original feature, and the scaled value is
the polygon's standing among all fitted province-years of the region.

READS
-----
    <deploy_dir>/_DEPLOY/deploy_manifest.json, estimator_deploy.pkl,
        scaler_deploy.pkl, copy_of_<config>.yaml (for pre_calc.XY)
    the XY cache that config names - the rows the deploy model was fitted on
    <deploy_dir>/_PROJ/<key>/estimators/<est>/X_for_<year>.csv
    <deploy_dir>/_PROJ/<key>/estimators/<est>/projection_for_<year>.csv

WRITES
------
    <deploy_dir>/_PROJ/<key>/shap_in_<year>_seed<seed>.csv
        long format, one row per polygon x feature: poly_ID, feature,
        value_raw, value_scaled, shap, base_value, score, n_emerging_rows,
        n_background

CHECKS - each guards a failure that would otherwise pass silently
------
  - the cache, after the same dropna as the fit, has the manifest's row count
    and the manifest's feature columns in order: a cache rebuilt after the fit
    would otherwise explain against a different population
  - re-scoring X_for_<year>.csv reproduces projection_for_<year>.csv: proves
    the matrix explained is the matrix that was scored
  - base_value + sum(shap) reproduces the score

Usage:
    python -m copro.scripts.explain <deploy_dir> --key proj_2026 --year 2026
"""

import argparse
import glob
import json
import os
import pickle

import click
import numpy as np
import pandas as pd

from copro import settings, xydata


def explain(deploy_dir, key, year, n_background=100, seed=42):
    """Writes the SHAP table for one projection key and year. Returns it."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "copro.scripts.explain needs shap, which is an optional extra.\n"
            "  pip install shap==0.49.1   (or: pip install copro[explain])"
        ) from exc
   
    deploy_dir = os.path.abspath(deploy_dir)
    out_dir_DEPLOY = os.path.join(deploy_dir, "_DEPLOY")

    with open(
        os.path.join(out_dir_DEPLOY, "deploy_manifest.json"), "r", encoding="utf-8"
    ) as f:
        manifest = json.load(f)
    feature_cols = manifest["feature_columns"]

    # same config lookup as project.py
    config_matches = sorted(glob.glob(os.path.join(out_dir_DEPLOY, "copy_of_*.yaml")))
    if len(config_matches) != 1:
        raise ValueError(
            f"Expected exactly one reference config copy in {out_dir_DEPLOY}, "
            f"found {len(config_matches)}: {[os.path.basename(f) for f in config_matches]}"
        )
    config = settings._parse_settings(config_matches[0])

    with open(os.path.join(out_dir_DEPLOY, manifest["estimator_file"]), "rb") as f:
        estimator = pickle.load(f)
    with open(os.path.join(out_dir_DEPLOY, manifest["scaler_file"]), "rb") as f:
        scaler = pickle.load(f)
    pos = list(estimator.classes_).index(1)

    click.echo(click.style("\nINFO: explain started\n", fg="cyan"))
    click.echo(f"Deploy model: {out_dir_DEPLOY}")
    click.echo(f"Feature set:  {manifest['feature_set']}")
    click.echo(f"Scenario:     {key}, {year}")

    # --- background: emerging rows of the fitted matrix ---------------------
    xy_cache = config.get("pre_calc", {}).get("XY", "")
    if not xy_cache:
        raise ValueError(
            "pre_calc.XY is empty in the deploy config "
            f"({os.path.basename(config_matches[0])}).\n"
            "explain.py needs the XY cache the deploy model was fitted on, to "
            "take its background rows from. Set pre_calc.XY to a cache path "
            "and refit the deploy model, or point it at the existing cache."
        )
    if not os.path.isfile(xy_cache):
        raise FileNotFoundError(
            f"XY cache named by pre_calc.XY does not exist: {xy_cache}"
        )
    click.echo(f"Reading fitted rows from XY cache {xy_cache}.")
    with open(xy_cache, "rb") as f:
        cached = pickle.load(f)
    # the same dropna the deploy fit went through
    X, _, meta = xydata._split_XY_data(cached["XY_df"])

    cache_cols = [str(c) for c in X.columns[2:]]
    if cache_cols != feature_cols or len(X) != manifest["n_rows"]:
        raise ValueError(
            "XY cache does not match the deploy fit - rebuilt since?\n"
            f"  rows:     cache {len(X)}, manifest {manifest['n_rows']}\n"
            f"  columns match manifest (in order): {cache_cols == feature_cols}"
        )

    emerging = meta["conflict_t_min_1"].to_numpy() == 0
    X_em = scaler.transform(X.iloc[:, 2:].to_numpy()[emerging])

    rng = np.random.default_rng(seed)
    n_bg = min(n_background, X_em.shape[0])
    background = X_em[rng.choice(X_em.shape[0], size=n_bg, replace=False)]
    # scikit-learn walks the trees in float32; round the same way so SHAP
    # routes every row exactly as predict_proba does
    background = background.astype(np.float32).astype(np.float64)    
    click.echo(
        f"Background: {n_bg} of {X_em.shape[0]} emerging rows "
        f"({len(X)} fitted rows in total), seed {seed}."
    )

    # --- the scored matrix ---------------------------------------------------
    est_dir = os.path.join(
        deploy_dir,
        "_PROJ",
        key,
        "estimators",
        manifest["estimator_file"].rsplit(".", maxsplit=1)[0],
    )
    X_proj = pd.read_csv(
        os.path.join(est_dir, f"X_for_{year}.csv"), float_precision="round_trip"
    )
    ids = X_proj["poly_ID"].astype(str)
    X_proj_ft = scaler.transform(X_proj[feature_cols].to_numpy())
    X_proj_ft = X_proj_ft.astype(np.float32).astype(np.float64)    
    score = estimator.predict_proba(X_proj_ft)[:, pos]

    proj = pd.read_csv(os.path.join(est_dir, f"projection_for_{year}.csv"))
    stored = proj.set_index(proj["ID"].astype(str))["y_prob_1"].astype(float)
    if set(ids) != set(stored.index) or not np.allclose(
        score, stored.loc[ids].to_numpy()
    ):
        raise ValueError(
            f"Re-scoring X_for_{year}.csv does not reproduce "
            f"projection_for_{year}.csv - the matrix is not the one scored."
        )

    # --- SHAP ----------------------------------------------------------------
    click.echo(f"Computing SHAP for {len(X_proj)} polygons x {len(feature_cols)} features.")
    explainer = shap.TreeExplainer(
        estimator, data=background, feature_perturbation="interventional"
    )

    # SHAP 0.49 caps the background at 100 rows whatever it is handed, so the
    # baseline is the mean over the rows it RETAINED, not over the n_background
    # sampled above. Are those retained rows actual background rows (a plain
    # subsample, so the baseline is an unbiased estimate of the emerging mean)
    # or synthetic summary points (so it is a different quantity)?
    kept = np.asarray(explainer.data)
    inset = sum(1 for r in kept if np.any(np.all(np.isclose(background, r), axis=1)))
    click.echo(
        f"Background retained by SHAP: {len(kept)} of {n_bg} handed in; "
        f"{inset} of {len(kept)} are actual background rows. "
        f"baseline {estimator.predict_proba(kept)[:, pos].mean():.8f} on retained, "
        f"{estimator.predict_proba(background)[:, pos].mean():.8f} on all handed in."
    )

    exp = explainer(X_proj_ft)
    values = exp.values[:, :, pos]
    base = exp.base_values[:, pos]

    # Tolerance admits the constant offset that the capped background leaves:
    # ~5e-5 on scores up to 0.83. The attributions reproduce each polygon's
    # difference from the baseline; it is the baseline that is a subsample mean.
    resid = base + values.sum(axis=1) - score
    tol = 1e-4 + 1e-4 * np.abs(score)
    bad = np.abs(resid) > tol
    click.echo(
        f"Additivity residual: max {np.abs(resid).max():.3e}, "
        f"mean {np.abs(resid).mean():.3e}, over tolerance in {bad.sum()} of {len(resid)} rows."
    )
    if bad.any():
        i = int(np.argmax(np.abs(resid)))
        click.echo(
            f"  worst: poly {ids.iloc[i]}, score {score[i]:.6f}, "
            f"base {base[i]:.6f}, sum(shap) {values[i].sum():.6f}, "
            f"resid {resid[i]:.3e}"
        )
        raise ValueError("base_value + sum(shap) does not reproduce the score.")
        
    n, k = values.shape
    out = pd.DataFrame(
        {
            "poly_ID": np.repeat(ids.to_numpy(), k),
            "feature": np.tile(feature_cols, n),
            "value_raw": X_proj[feature_cols].astype(float).to_numpy().ravel(),
            "value_scaled": X_proj_ft.ravel(),
            "shap": values.ravel(),
            "base_value": np.repeat(base, k),
            "score": np.repeat(score, k),
            "n_emerging_rows": X_em.shape[0],
            "n_background": n_bg,
        }
    )

    out_file = os.path.join(deploy_dir, "_PROJ", key, f"shap_in_{year}_seed{seed}.csv")
    out.to_csv(out_file, index=False)
    click.echo(f"Wrote {out_file}")
    click.echo(click.style("\nINFO: explain finished\n", fg="cyan"))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="SHAP attribution of a CoPro deploy projection."
    )
    ap.add_argument("deploy_dir", help="the deploy folder holding _DEPLOY and _PROJ")
    ap.add_argument("--key", required=True, help="projection key, e.g. proj_2026")
    ap.add_argument("--year", type=int, required=True, help="projection year")
    ap.add_argument(
        "--n-background",
        type=int,
        default=100,
        help="emerging rows sampled as background (all if fewer). Default 100: "
             "SHAP 0.49 caps the background at 100 rows whatever it is handed, "
             "so anything larger is sampled and then discarded.",
    )
    ap.add_argument("--seed", type=int, default=42, help="background sample seed.")
    args = ap.parse_args()

    explain(
        deploy_dir=args.deploy_dir,
        key=args.key,
        year=args.year,
        n_background=args.n_background,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
