#!/usr/bin/env python
"""
Grouped permutation importance on the EMERGING subset.
(CoPro postprocessing - run after a region sweep.)

Standalone diagnostic - does NOT modify the CoPro pipeline. For each region it
re-fits an RF from the cached indicator matrix (XY.gpkg), restricts the TEST set
to emerging rows (conflict_t_min_1 == 0), and measures how much emerging Average
Precision drops when each indicator GROUP is jointly permuted. Groups come from
indicator_categories.csv (domain, type).

The temporal split is a RULE, not a draw: it holds out the same years however
many times it is run, so a single temporal panel is one unreplicated
measurement with no spread. The cutoff is therefore swept over a narrow window
(TEMPORAL_CUTOFFS) and each cutoff written as its own set of rows. Read the
spread ACROSS cutoffs as the floor below which a temporal difference means
nothing; within one cutoff, ap_drop_sd is permutation noise only.

The window is deliberately narrow. Widening it shortens the test period and
drops the emerging-onset count, and that variation would dominate any
difference between cutoffs.

Cutoffs are applied at SPLIT time to the XY's own `year` column, so the XY
cache does not depend on the cutoff - no rebuild, no cache delete.

Reads : <OUTPUT_ROOT>/<sim>/_REF/XY.gpkg  +  indicator_categories.csv
Writes: <OUTPUT_ROOT>/eval/summary/grouped_importance.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import preprocessing, model_selection, ensemble
from sklearn.metrics import average_precision_score

from copro import settings

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
OUTPUT_ROOT = "C:/Siderius_UNCHARTED/Projects/2026 FAO conflict II/4.Analysis/copro/output"
CATEGORIES_FILE = "./workflows/indicator_categories.csv"

STRATEGIES = ["random", "temporal", "leave_polygons_out"]

# Standalone fallback only - the R side passes --temporal-cutoff, and the
# wrapper's TEMPORAL_CUTOFF / TEMPORAL_CUTOFF_WINDOW are the source of truth.
# Same two names here so the fallback follows the centre year rather than
# pinning a window that quietly stops matching the pipeline.
TEMPORAL_CUTOFF = 2017
TEMPORAL_CUTOFF_WINDOW = 3
if TEMPORAL_CUTOFF_WINDOW % 2 == 0:
    raise ValueError(
        "TEMPORAL_CUTOFF_WINDOW must be odd - the window is centred on "
        "TEMPORAL_CUTOFF"
    )
TEMPORAL_CUTOFFS = [
    TEMPORAL_CUTOFF + d
    for d in range(-(TEMPORAL_CUTOFF_WINDOW // 2), TEMPORAL_CUTOFF_WINDOW // 2 + 1)
]

SEED = 42
TRAIN_FRACTION = 0.7
N_REPEATS = 10
N_JOBS = 6

# fixed RF config: the package default, read from settings rather than typed
# here. Holding it fixed across arms AND cutoffs is the
# point: a difference is then attributable to features or to the cutoff. This
# script reads no region config, so it always uses the package default - if a
# run overrode hyperparameters in its config or switched the search on, these
# AP values are a common yardstick and not that run's configuration. They are
# not any region's deployed onset skill either way, and should not be quoted
# as such.
RF_PARAMS = dict(settings.DEFAULT_HYPERPARAMETERS)

# columns in XY.gpkg that are NOT model features
NON_FEATURES = {"poly_ID", "year", "conflict", "geometry", "poly_geometry"}


# --------------------------------------------------------------------------- #
# helpers (flat)
# --------------------------------------------------------------------------- #
def make_split(strategy, meta, n, seed, cutoff=None):
    """Returns (train_idx, test_idx) for the chosen strategy.

    `cutoff` is required for 'temporal' and ignored by the other strategies.
    """
    if strategy == "random":
        return model_selection.train_test_split(
            np.arange(n), test_size=1 - TRAIN_FRACTION, random_state=seed
        )
    if strategy == "temporal":
        if cutoff is None:
            raise ValueError("strategy 'temporal' requires a cutoff year")
        yrs = meta["year"].to_numpy()
        tr = np.where(yrs <= cutoff)[0]
        te = np.where(yrs > cutoff)[0]
        if len(tr) == 0 or len(te) == 0:
            raise ValueError(
                f"temporal cutoff {cutoff} leaves an empty train or test set "
                f"(XY years {int(yrs.min())}-{int(yrs.max())})."
            )
        return tr, te
    if strategy == "leave_polygons_out":
        gss = model_selection.GroupShuffleSplit(
            n_splits=1, test_size=1 - TRAIN_FRACTION, random_state=seed
        )
        return next(gss.split(np.arange(n), groups=meta["poly_ID"].to_numpy()))
    raise ValueError(f"unknown strategy {strategy}")


def build_jobs(strategies, temporal_cutoffs):
    """Expand strategies into (strategy, cutoff) jobs.

    Only 'temporal' is swept - random and leave_polygons_out already vary
    their test set with the seed, and carry cutoff = None.
    """
    jobs = []
    for strategy in strategies:
        if strategy == "temporal":
            for cutoff in temporal_cutoffs:
                jobs.append((strategy, cutoff))
        else:
            jobs.append((strategy, None))
    return jobs


def build_groups(cats, feature_cols):
    """Build a dict {group_name: [columns]} from the categories table.
    Three views: each domain, agri-food split by type, and the type axis.
    """
    cats = cats[cats["variable"].isin(feature_cols)]
    groups = {}
    for dom, sub in cats.groupby("domain"):
        groups[f"domain:{dom}"] = sub["variable"].tolist()
    af = cats[cats["domain"] == "agri_food"]
    for t, sub in af.groupby("type"):
        groups[f"agri_food:{t}"] = sub["variable"].tolist()
    for t, sub in cats.groupby("type"):
        groups[f"type:{t}"] = sub["variable"].tolist()
    return groups


def grouped_perm_importance(rf, X_em, y_em, base_ap, col_index, cols, rng):
    """Joint-permute the block `cols` in X_em and return mean/sd AP drop."""
    idx = [col_index[c] for c in cols]
    drops = np.empty(N_REPEATS)
    for r in range(N_REPEATS):
        Xp = X_em.copy()
        perm = rng.permutation(X_em.shape[0])
        Xp[:, idx] = Xp[perm][:, idx]  # same row shuffle across all block columns
        ap = average_precision_score(y_em, rf.predict_proba(Xp)[:, 1])
        drops[r] = base_ap - ap
    return drops.mean(), drops.std()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(output_root=OUTPUT_ROOT, categories_file=CATEGORIES_FILE, region=None,
         source=None, temporal_cutoffs=TEMPORAL_CUTOFFS):
    cats = pd.read_csv(categories_file)
    out_dir = os.path.join(output_root, "eval", "summary")
    os.makedirs(out_dir, exist_ok=True)

    jobs = build_jobs(STRATEGIES, temporal_cutoffs)
    print(f"Temporal cutoffs: {temporal_cutoffs}")

    if region and source:
        pattern = os.path.join(output_root, f"{region}_{source.lower()}", "_REF", "XY.gpkg")
    elif region:
        pattern = os.path.join(output_root, f"{region}_*", "_REF", "XY.gpkg")
    else:
        pattern = os.path.join(output_root, "*", "_REF", "XY.gpkg")
    xy_paths = sorted(glob.glob(pattern))
    if not xy_paths:
        raise FileNotFoundError(f"no XY.gpkg found matching {pattern}")

    records = []
    for xy_path in xy_paths:
        sim = os.path.basename(os.path.dirname(os.path.dirname(xy_path)))
        print(f"\n=== {sim} ===")

        df = gpd.read_file(xy_path)
        if "geometry" in df.columns:
            df = pd.DataFrame(df.drop(columns="geometry"))

        required = {"year", "conflict", "conflict_t_min_1", "poly_ID"}
        miss = required - set(df.columns)
        if miss:
            print(f"  SKIP {sim}: XY.gpkg missing {sorted(miss)} "
                  f"(stale, pre-year-column run?)")
            continue

        feature_cols = [c for c in df.columns if c not in NON_FEATURES]
        missing = set(feature_cols) - set(cats["variable"])
        if missing:
            print(f"  WARNING: features not in categories (won't be grouped): {sorted(missing)}")

        # drop rows with any missing feature or label (matches the model)
        df = df.dropna(subset=feature_cols + ["conflict"]).reset_index(drop=True)

        X = df[feature_cols].to_numpy(dtype=float)
        Y = df["conflict"].astype(int).to_numpy()
        meta = df[["year", "poly_ID", "conflict_t_min_1"]].copy()
        col_index = {c: i for i, c in enumerate(feature_cols)}
        groups = build_groups(cats, feature_cols)

        for strategy, cutoff in jobs:
            tag = strategy if cutoff is None else f"{strategy}@{cutoff}"
            tr, te = make_split(strategy, meta, len(df), SEED, cutoff)

            scaler = preprocessing.QuantileTransformer(random_state=42)
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])

            rf = ensemble.RandomForestClassifier(n_jobs=N_JOBS, **RF_PARAMS)
            rf.fit(X_tr, Y[tr])

            # restrict test set to emerging rows
            em = meta.iloc[te]["conflict_t_min_1"].to_numpy() == 0
            X_em = X_te[em]
            y_em = Y[te][em]
            if y_em.sum() == 0 or y_em.sum() == len(y_em):
                print(f"  {tag}: no emerging onset variation in test - skipped")
                continue

            base_ap = average_precision_score(y_em, rf.predict_proba(X_em)[:, 1])
            rng = np.random.default_rng(SEED)
            print(f"  {tag}: train n={len(tr)} emerging n={len(y_em)} "
                  f"pos={int(y_em.sum())} base AP={base_ap:.3f}")

            for gname, cols in groups.items():
                mean_drop, sd_drop = grouped_perm_importance(
                    rf, X_em, y_em, base_ap, col_index, cols, rng
                )
                records.append({
                    "region_source": sim,
                    "strategy": strategy,
                    "cutoff": cutoff if cutoff is not None else "",
                    "group": gname,
                    "n_features": len(cols),
                    "base_ap": round(base_ap, 4),
                    "ap_drop": round(mean_drop, 4),
                    "ap_drop_sd": round(sd_drop, 4),
                    "emerging_pos": int(y_em.sum()),
                })

    res = pd.DataFrame(records)
    out_file = os.path.join(out_dir, "grouped_importance.csv")
    res.to_csv(out_file, index=False)
    print(f"\nWrote {out_file}  ({len(res)} rows)")

    # console summary: domain importance on the honest splits. Temporal is
    # printed once per cutoff - pooling them would average away the spread the
    # sweep exists to show.
    show = res[res["group"].str.startswith("domain:") |
               res["group"].str.startswith("agri_food:")]
    panels = [("temporal", c) for c in temporal_cutoffs]
    panels.append(("leave_polygons_out", None))
    for strat, cutoff in panels:
        sub = show[show["strategy"] == strat]
        if cutoff is not None:
            sub = sub[sub["cutoff"] == cutoff]
        if sub.empty:
            continue
        tag = strat if cutoff is None else f"{strat} @ {cutoff}"
        print(f"\n==== AP drop by group ({tag}) ====")
        piv = sub.pivot_table(index="group", columns="region_source",
                              values="ap_drop").round(3)
        print(piv.to_string())

    # emerging-onset count per temporal cutoff: if this moves a lot between
    # cutoffs, the spread is base-rate variation rather than cutoff
    # sensitivity, and the window is too wide.
    tmp = res[res["strategy"] == "temporal"]
    if not tmp.empty:
        print("\n==== emerging positives by cutoff ====")
        piv = tmp.pivot_table(index="cutoff", columns="region_source",
                              values="emerging_pos", aggfunc="first")
        print(piv.to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Grouped permutation importance on emerging onset.")
    ap.add_argument("--output-root", default=OUTPUT_ROOT, help="CoPro output root (contains <sim>/_REF and eval/)")
    ap.add_argument("--categories", default=CATEGORIES_FILE, help="path to indicator_categories.csv")
    ap.add_argument("--region", default=None, help="optional: restrict to one region (sim folder prefix)")
    ap.add_argument("--source", default=None, help="optional: conflict source suffix, e.g. ucdp")
    ap.add_argument("--temporal-cutoff", type=int, nargs="+", default=TEMPORAL_CUTOFFS,
                    dest="temporal_cutoffs",
                    help="one or more temporal split cutoff years (default: %(default)s)")
    args = ap.parse_args()
    main(output_root=args.output_root, categories_file=args.categories,
         region=args.region, source=args.source,
         temporal_cutoffs=args.temporal_cutoffs)
