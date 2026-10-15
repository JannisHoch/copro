#!/usr/bin/env python
"""
Emerging-only onset model experiment (pooled vs onset vs blend).
(CoPro postprocessing - run after a region sweep.)

Standalone diagnostic - does NOT modify the CoPro pipeline. For each region it
re-fits TWO RFs from the cached indicator matrix (XY.gpkg) on a SHARED split:

  pooled : trained on all training rows, all features (incl. conflict_t_min_1)
  onset  : trained on peaceful training rows only (conflict_t_min_1 == 0),
           with the conflict_t_min_1 column dropped (always 0 on those rows)
           and conflict_t_min_1_nb (spillover) retained

Both are evaluated on the SAME emerging test rows (conflict_t_min_1 == 0).
The blend  w * p_onset + (1 - w) * p_pooled  is swept over w = 0.0 .. 1.0:
w = 0 is the pooled reference, w = 1 the standalone onset model, so one table
covers the pooled model, the onset model and every blend in between.

Splits are made on the FULL dataset first (random / temporal /
leave_polygons_out, identical logic to grouped_importance.py), so held-out
years/polygons are held out for both models. random and leave_polygons_out are
repeated N_RUNS times with seed SEED + run (per the validation discipline);
temporal is deterministic and run once.

Two mechanism checks ride along per run:
  check 1: Spearman rank correlation between p_pooled and p_onset on the
           emerging test rows (high agreement = same ranking, blend has
           nothing to exploit)
  check 2: depths at which the POOLED forest splits on conflict_t_min_1
           (early first use = strata partitioned near the root, the peaceful
           branch is an embedded onset model)
  check 3: which features take the upper-tree splits (depth <= 2) of the
           pooled forest - names the substitutes that partition the data
           when conflict_t_min_1 is not chosen (candidate targets for
           converting level features to anomalies)

Reads : <output_root>/<sim>/_REF/XY.gpkg
Writes: <output_root>/eval/summary/onset_experiment.csv
        <output_root>/eval/summary/onset_experiment_tmin1_depths.csv
        <output_root>/eval/summary/onset_experiment_top_splits.csv

output_root is one run's output root and temporal_cutoff the last training
year of the temporal split. Both are required arguments, with no default here,
so the experiment cannot silently run against another run or another cutoff
than intended.
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from sklearn import preprocessing, model_selection, ensemble
from sklearn.metrics import average_precision_score

from copro import settings

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
REGIONS = None                            # sim folder prefixes; None = all sims
SOURCE = "ucdp"                           # conflict source suffix; None = all

STRATEGIES = ["random", "temporal", "leave_polygons_out"]
SEED = 42
TRAIN_FRACTION = 0.7
N_RUNS = 10            # random / leave_polygons_out; temporal runs once
N_JOBS = 6

BLEND_WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 1)

# fixed RF config for BOTH models: the package default, read from settings
# (same as grouped_importance.py), so the comparison is about
# the training population, not hyperparameters. NOTE: class_weight {1: 100}
# was chosen for the pooled problem - flagged for a second look if onset
# results are marginal.
RF_PARAMS = dict(settings.DEFAULT_HYPERPARAMETERS)

# columns in XY.gpkg that are NOT model features
NON_FEATURES = {"poly_ID", "year", "conflict", "geometry", "poly_geometry"}


# --------------------------------------------------------------------------- #
# helpers (flat)
# --------------------------------------------------------------------------- #
def make_split(strategy, meta, n, seed, temporal_cutoff):
    """Returns (train_idx, test_idx) for the chosen strategy."""
    if strategy == "random":
        return model_selection.train_test_split(
            np.arange(n), test_size=1 - TRAIN_FRACTION, random_state=seed
        )
    if strategy == "temporal":
        yrs = meta["year"].to_numpy()
        tr = np.where(yrs <= temporal_cutoff)[0]
        te = np.where(yrs > temporal_cutoff)[0]
        return tr, te
    if strategy == "leave_polygons_out":
        gss = model_selection.GroupShuffleSplit(
            n_splits=1, test_size=1 - TRAIN_FRACTION, random_state=seed
        )
        return next(gss.split(np.arange(n), groups=meta["poly_ID"].to_numpy()))
    raise ValueError(f"unknown strategy {strategy}")


def walk_pooled_forest(rf, tmin1_col, top_depth=2):
    """Single pass over the pooled forest for checks 2 and 3.
    Returns:
      all_depths : depths of every split on conflict_t_min_1
      first_use  : first conflict_t_min_1 split depth per tree (trees that
                   never use it are omitted - their count follows from
                   n_estimators - len(first_use))
      top_counts : {(depth, feature_idx): n_splits} for all splits at
                   depth <= top_depth (which features take the upper tree)
    Leaf nodes have feature == -2, so they never match a feature index."""
    all_depths = []
    first_use = []
    top_counts = {}
    for est in rf.estimators_:
        t = est.tree_
        depth = np.zeros(t.node_count, dtype=int)
        node_stack = [0]
        while node_stack:
            node = node_stack.pop()
            for child in (t.children_left[node], t.children_right[node]):
                if child != -1:
                    depth[child] = depth[node] + 1
                    node_stack.append(child)
        feats = t.feature
        d_t = depth[feats == tmin1_col]
        if len(d_t):
            all_depths.extend(d_t.tolist())
            first_use.append(int(d_t.min()))
        for node in np.where((feats >= 0) & (depth <= top_depth))[0]:
            key = (int(depth[node]), int(feats[node]))
            top_counts[key] = top_counts.get(key, 0) + 1
    return np.array(all_depths), np.array(first_use), top_counts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(output_root, temporal_cutoff, regions=REGIONS, source=SOURCE,
         n_runs=N_RUNS):

    out_dir = os.path.join(output_root, "eval", "summary")
    os.makedirs(out_dir, exist_ok=True)

    # discover region sims with a cached XY.gpkg
    xy_files = sorted(glob.glob(os.path.join(output_root, "*", "_REF", "XY.gpkg")))
    if not xy_files:
        raise FileNotFoundError(f"no <sim>/_REF/XY.gpkg found under {output_root}")

    records = []
    depth_records = []
    top_records = []

    for xy_file in xy_files:
        sim = os.path.basename(os.path.dirname(os.path.dirname(xy_file)))
        if regions is not None and not any(sim.startswith(r) for r in regions):
            continue
        if source is not None and not sim.endswith(source):
            continue

        print(f"\n=== {sim} ===")
        df = gpd.read_file(xy_file)

        feature_cols = [c for c in df.columns if c not in NON_FEATURES]
        if "conflict_t_min_1" not in feature_cols:
            raise ValueError(f"{sim}: conflict_t_min_1 not among features - wrong XY.gpkg?")
        onset_cols = [c for c in feature_cols if c != "conflict_t_min_1"]
        tmin1_col = feature_cols.index("conflict_t_min_1")

        # drop rows with any missing feature or label (matches the model)
        df = df.dropna(subset=feature_cols + ["conflict"]).reset_index(drop=True)

        X = df[feature_cols].to_numpy(dtype=float)
        X_on = df[onset_cols].to_numpy(dtype=float)
        Y = df["conflict"].astype(int).to_numpy()
        meta = df[["year", "poly_ID", "conflict_t_min_1"]].copy()
        peaceful = meta["conflict_t_min_1"].to_numpy() == 0

        for strategy in STRATEGIES:
            runs = 1 if strategy == "temporal" else n_runs

            for run in range(runs):
                tr, te = make_split(strategy, meta, len(df), SEED + run,
                                    temporal_cutoff)

                # emerging test rows (shared evaluation set for both models)
                em_te = te[peaceful[te]]
                y_em = Y[em_te]
                if y_em.sum() == 0 or y_em.sum() == len(y_em):
                    print(f"  {strategy} run {run}: no emerging onset variation "
                          f"in test - skipped")
                    continue

                # onset-model training rows: peaceful training rows only
                tr_on = tr[peaceful[tr]]
                if Y[tr_on].sum() == 0:
                    print(f"  WARNING: {strategy} run {run}: no onset positives "
                          f"in peaceful training rows - onset model skipped")
                    continue

                # pooled model (all training rows, all features)
                scaler_p = preprocessing.QuantileTransformer(random_state=42)
                Xp_tr = scaler_p.fit_transform(X[tr])
                Xp_em = scaler_p.transform(X[em_te])
                rf_p = ensemble.RandomForestClassifier(n_jobs=N_JOBS, **RF_PARAMS)
                rf_p.fit(Xp_tr, Y[tr])
                p_pooled = rf_p.predict_proba(Xp_em)[:, 1]

                # onset model (peaceful training rows, conflict_t_min_1 dropped)
                scaler_o = preprocessing.QuantileTransformer(random_state=42)
                Xo_tr = scaler_o.fit_transform(X_on[tr_on])
                Xo_em = scaler_o.transform(X_on[em_te])
                rf_o = ensemble.RandomForestClassifier(n_jobs=N_JOBS, **RF_PARAMS)
                rf_o.fit(Xo_tr, Y[tr_on])
                p_onset = rf_o.predict_proba(Xo_em)[:, 1]

                base_rate = y_em.mean()
                ap_w0 = average_precision_score(y_em, p_pooled)
                ap_w1 = average_precision_score(y_em, p_onset)

                # check 1: rank agreement of the two models on emerging rows
                rho = stats.spearmanr(p_pooled, p_onset)[0]

                # checks 2 + 3: conflict_t_min_1 split depths, and which
                # features take the upper-tree splits (single forest walk)
                all_d, first_d, top_counts = walk_pooled_forest(rf_p, tmin1_col)
                med_first = np.median(first_d) if len(first_d) else np.nan
                for d in np.unique(all_d):
                    depth_records.append({
                        "region_source": sim,
                        "strategy": strategy,
                        "run": run,
                        "depth": int(d),
                        "n_splits": int((all_d == d).sum()),
                        "n_trees_first_use": int((first_d == d).sum()),
                        "n_trees_using": len(first_d),
                        "n_trees_total": rf_p.n_estimators,
                    })
                for (d, fidx), cnt in sorted(top_counts.items()):
                    top_records.append({
                        "region_source": sim,
                        "strategy": strategy,
                        "run": run,
                        "depth": d,
                        "feature": feature_cols[fidx],
                        "n_splits": cnt,
                    })

                print(f"  {strategy} run {run}: emerging n={len(y_em)} "
                      f"pos={int(y_em.sum())} | pooled AP={ap_w0:.3f} "
                      f"onset AP={ap_w1:.3f} | spearman={rho:.3f} | "
                      f"t-1 first split: median depth {med_first:.0f}, "
                      f"{len(first_d)}/{rf_p.n_estimators} trees")

                for w in BLEND_WEIGHTS:
                    ap = average_precision_score(
                        y_em, w * p_onset + (1 - w) * p_pooled
                    )
                    records.append({
                        "region_source": sim,
                        "strategy": strategy,
                        "run": run,
                        "w": w,
                        "emerging_ap": round(ap, 4),
                        "emerging_n": len(y_em),
                        "emerging_pos": int(y_em.sum()),
                        "base_rate": round(base_rate, 4),
                        "train_n_onset": len(tr_on),
                        "train_pos_onset": int(Y[tr_on].sum()),
                        "spearman_pooled_onset": round(rho, 4),
                    })

    if not records:
        raise RuntimeError("no results produced - check REGIONS/SOURCE filters")

    res = pd.DataFrame(records)
    out_file = os.path.join(out_dir, "onset_experiment.csv")
    res.to_csv(out_file, index=False)
    print(f"\nWrote {out_file}  ({len(res)} rows)")

    # console summary: mean AP (over runs) per weight, honest splits
    for strat in ["temporal", "leave_polygons_out"]:
        sub = res[res["strategy"] == strat]
        if sub.empty:
            continue
        print(f"\n==== emerging AP by blend weight ({strat}, mean over runs) ====")
        piv = sub.pivot_table(index="w", columns="region_source",
                              values="emerging_ap", aggfunc="mean").round(3)
        print(piv.to_string())

    # console summary: pooled vs onset vs best blend, mean +/- sd over runs
    print("\n==== pooled (w=0) vs onset (w=1) vs best blend ====")
    for (sim, strat), sub in res.groupby(["region_source", "strategy"]):
        by_w = sub.groupby("w")["emerging_ap"]
        mean_w, sd_w = by_w.mean(), by_w.std()
        w_best = mean_w.idxmax()
        print(f"  {sim} | {strat}: "
              f"pooled {mean_w[0.0]:.3f}±{0.0 if np.isnan(sd_w[0.0]) else sd_w[0.0]:.3f} | "
              f"onset {mean_w[1.0]:.3f}±{0.0 if np.isnan(sd_w[1.0]) else sd_w[1.0]:.3f} | "
              f"best w={w_best:.1f} -> {mean_w[w_best]:.3f}")

    # check 1 summary: rank agreement pooled vs onset on emerging rows
    print("\n==== check 1: Spearman(p_pooled, p_onset) on emerging rows "
          "(mean over runs) ====")
    sp = res.groupby(["region_source", "strategy"])["spearman_pooled_onset"] \
            .mean().round(3)
    print(sp.to_string())

    # check 2 summary: how early the pooled forest uses conflict_t_min_1
    if depth_records:
        dep = pd.DataFrame(depth_records)
        depth_file = os.path.join(out_dir, "onset_experiment_tmin1_depths.csv")
        dep.to_csv(depth_file, index=False)
        print(f"\nWrote {depth_file}  ({len(dep)} rows)")

        print("\n==== check 2: share of trees whose FIRST conflict_t_min_1 "
              "split is at depth <= 3 ====")
        dep["first_le3"] = np.where(dep["depth"] <= 3,
                                    dep["n_trees_first_use"], 0)
        agg = dep.groupby(["region_source", "strategy"])[
            ["first_le3", "n_trees_first_use"]].sum()
        agg["share_first_le3"] = (agg["first_le3"]
                                  / agg["n_trees_first_use"]).round(3)
        print(agg[["share_first_le3"]].to_string())

    # check 3 summary: which features take the upper-tree splits
    if top_records:
        top = pd.DataFrame(top_records)
        top_file = os.path.join(out_dir, "onset_experiment_top_splits.csv")
        top.to_csv(top_file, index=False)
        print(f"\nWrote {top_file}  ({len(top)} rows)")

        print("\n==== check 3: features taking the upper-tree splits "
              "(share of splits at depth <= 2; root-only share in brackets; "
              "aggregated over strategies and runs) ====")
        for sim, sub in top.groupby("region_source"):
            total = sub["n_splits"].sum()
            root = sub[sub["depth"] == 0]
            root_total = root["n_splits"].sum()
            feat_agg = sub.groupby("feature")["n_splits"] \
                          .sum().sort_values(ascending=False)
            root_agg = root.groupby("feature")["n_splits"].sum()
            print(f"\n  {sim}:")
            for feat in feat_agg.index[:10]:
                share = feat_agg[feat] / total
                root_share = root_agg.get(feat, 0) / root_total
                print(f"    {feat:<40s} {share:6.3f}  (root {root_share:.3f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Emerging-only onset model vs pooled model vs blend."
    )
    ap.add_argument("--output-root", required=True,
                    help="one run's output root (contains <sim>/_REF and eval/)")
    ap.add_argument("--temporal-cutoff", type=int, required=True,
                    help="last training year of the temporal split")
    ap.add_argument("--regions", default="all",
                    help="comma-separated sim folder prefixes; 'all' = no filter")
    ap.add_argument("--source", default=SOURCE,
                    help="conflict source suffix, e.g. ucdp; 'all' = no filter")
    ap.add_argument("--n-runs", type=int, default=N_RUNS,
                    help="runs for random/leave_polygons_out (temporal always 1)")
    args = ap.parse_args()
    regions = None if args.regions == "all" else args.regions.split(",")
    source = None if args.source == "all" else args.source
    main(output_root=args.output_root, temporal_cutoff=args.temporal_cutoff,
         regions=regions, source=source, n_runs=args.n_runs)
