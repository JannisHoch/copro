"""CoPro output plotting - standard plots for a first look at a finished run.

Standalone entry point for inspecting CoPro output without the R workflow.
Uses CoPro's built-in `plots` module where it provides something useful.

    python plot_copro_outputs.py --output-root ../OUT                       # list available runs
    python plot_copro_outputs.py MENA --output-root ../OUT                  # show plots on screen
    python plot_copro_outputs.py MENA --output-root ../OUT --save           # write PNGs to the run dir
    python plot_copro_outputs.py MENA --output-root ../OUT --source acled   # non-default conflict source

--output-root is required, as in grouped_importance.py and onset_experiment.py,
so the plots cannot silently be made from another run's output.
"""

import argparse
import sys
from pathlib import Path

# non-interactive backend must be selected before pyplot is imported
import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from copro import plots
from copro.xydata import CONFLICT_DERIVED_KEYS

plt.ioff()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Optional. Two columns are used if present: `variable` and `label`. When the
# file is missing, raw variable names are used and nothing breaks. Keeping the
# labels in one CSV avoids a second copy drifting out of step with the R side.
INDICATOR_CATEGORIES_FILE = Path("indicator_categories.csv")

DEFAULT_SOURCE = "ucdp"

# Columns in XY.gpkg that are not indicators. CONFLICT_DERIVED_KEYS is imported
# rather than restated so that adding a conflict-history feature in xydata.py
# does not silently leak it into the correlation matrix here.
NON_INDICATOR_COLS = ["poly_ID", "poly_geometry", "geometry", "year", "conflict"]
NON_INDICATOR_COLS += CONFLICT_DERIVED_KEYS


# -----------------------------------------------------------------------------
# Paths and readers
# -----------------------------------------------------------------------------


def get_output_path(region, source=DEFAULT_SOURCE, output_root=None):
    """Returns the _REF directory for a region.

    Run directories are named `<region>_<source>`. A bare `<region>` directory
    is accepted as a fallback for older output trees.
    """

    root = Path(output_root)

    for name in (f"{region}_{source}", region):
        path = root / name / "_REF"
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No _REF folder for region '{region}' (source '{source}') under {root}. "
        f"Tried {region}_{source}/_REF and {region}/_REF."
    )


def list_runs(output_root=None):
    """Returns (region, source) pairs for every run directory found."""

    root = Path(output_root)

    runs = []
    for d in sorted(root.iterdir()):
        if not (d.is_dir() and (d / "_REF").exists()):
            continue
        region, _, source = d.name.rpartition("_")
        runs.append((region, source) if region else (d.name, DEFAULT_SOURCE))
    return runs


def read_importances(region, source=DEFAULT_SOURCE, output_root=None):
    return pd.read_parquet(get_output_path(region, source, output_root) / "perm_importances.parquet")


def read_metrics(region, source=DEFAULT_SOURCE, output_root=None):
    return pd.read_csv(get_output_path(region, source, output_root) / "evaluation_metrics.csv")


def read_predictions(region, source=DEFAULT_SOURCE, output_root=None):
    return gpd.read_file(get_output_path(region, source, output_root) / "output_for_REF.gpkg")


def read_xy_data(region, source=DEFAULT_SOURCE, output_root=None):
    return gpd.read_file(get_output_path(region, source, output_root) / "XY.gpkg")


def read_selected_polygons(region, source=DEFAULT_SOURCE, output_root=None):
    return gpd.read_file(get_output_path(region, source, output_root) / "selected_polygons.gpkg")


def read_selected_conflicts(region, source=DEFAULT_SOURCE, output_root=None):
    return gpd.read_file(get_output_path(region, source, output_root) / "selected_conflicts.gpkg")


def _load_labels():
    """Returns a {variable: label} mapping, empty if the categories file is absent."""

    if not INDICATOR_CATEGORIES_FILE.exists():
        return {}
    cats = pd.read_csv(INDICATOR_CATEGORIES_FILE)
    if not {"variable", "label"}.issubset(cats.columns):
        return {}
    return dict(zip(cats["variable"], cats["label"]))


LABELS = _load_labels()


def apply_labels(var_names):
    return [LABELS.get(v, v) for v in var_names]


def _finish(fig_path, save):
    if save:
        plt.savefig(fig_path, dpi=150)
        print(f"  saved: {fig_path.name}")
    else:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_polygons(region, source=DEFAULT_SOURCE, save=False, output_root=None):
    """Selected polygon boundaries."""

    polygons = read_selected_polygons(region, source, output_root)

    fig, ax = plt.subplots(figsize=(12, 10))
    plots.selected_polygons(polygons, ax=ax, color="black", linewidth=0.5)
    ax.set_title(f"Selected polygons: {region}")
    ax.set_axis_off()
    plt.tight_layout()
    _finish(get_output_path(region, source, output_root) / "py_polygons.png", save)


def plot_conflicts(region, source=DEFAULT_SOURCE, save=False, output_root=None):
    """Selected conflict events, over polygon boundaries where available."""

    conflicts = read_selected_conflicts(region, source, output_root)

    fig, ax = plt.subplots(figsize=(12, 10))

    try:
        read_selected_polygons(region, source, output_root).boundary.plot(
            ax=ax, color="gray", linewidth=0.3
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"    (no polygon background: {e})")

    color_col = next(
        (c for c in ("best", "fatalities", "deaths", "casualties") if c in conflicts.columns),
        None,
    )
    if color_col:
        conflicts.plot(
            ax=ax, column=color_col, cmap="Reds", legend=True, markersize=5, alpha=0.7
        )
    else:
        conflicts.plot(ax=ax, color="red", markersize=5, alpha=0.7)

    ax.set_title(f"Selected conflicts: {region}")
    ax.set_axis_off()
    plt.tight_layout()
    _finish(get_output_path(region, source, output_root) / "py_conflicts.png", save)


def plot_metrics_distribution(region, source=DEFAULT_SOURCE, save=False, output_root=None):
    """Distribution of evaluation metrics across runs."""

    metrics_df = read_metrics(region, source, output_root)

    metric_cols = [
        c
        for c in metrics_df.columns
        if c not in ("run", "X", "Unnamed: 0", "index")
        and pd.api.types.is_numeric_dtype(metrics_df[c])
    ]

    out_dict = {col: metrics_df[col].tolist() for col in metric_cols}

    ax = plots.metrics_distribution(out_dict, metric_cols, figsize=(10, 6))
    ax.set_title(f"Metrics distribution: {region} ({len(metrics_df)} runs)")
    ax.set_xlabel("Score")
    plt.tight_layout()
    _finish(get_output_path(region, source, output_root) / "py_metrics_distribution.png", save)


def plot_correlation(region, source=DEFAULT_SOURCE, save=False, output_root=None):
    """Correlation matrix over indicators only.

    Conflict-history columns are excluded. They are all derived from the same
    conflict lookup, so leaving them in puts a near-perfectly correlated block
    in a matrix that is meant to show indicator collinearity.
    """

    xy = read_xy_data(region, source, output_root)

    indicator_cols = [
        c
        for c in xy.columns
        if c not in NON_INDICATOR_COLS
        and not c.startswith("poly_")
        and pd.api.types.is_numeric_dtype(xy[c])
    ]

    cor_matrix = pd.DataFrame(xy[indicator_cols]).corr()
    labels = apply_labels(cor_matrix.columns.tolist())

    # imshow rather than seaborn: the seaborn heatmap renders empty in some
    # backends used here.
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cor_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(cor_matrix)):
        for j in range(len(cor_matrix)):
            val = cor_matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=7,
            )

    ax.set_title(f"Indicator correlation: {region}")
    plt.tight_layout()
    _finish(get_output_path(region, source, output_root) / "py_correlation.png", save)


def plot_importance(region, source=DEFAULT_SOURCE, save=False, output_root=None):
    """Mean permutation importance per variable."""

    importances = read_importances(region, source, output_root)
    mean_imp = importances.mean(numeric_only=True).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(mean_imp))
    ax.barh(y_pos, mean_imp.values, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(apply_labels(mean_imp.index.tolist()))
    ax.set_xlabel("Mean permutation importance")
    ax.set_title(f"Variable importance: {region}")
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    _finish(get_output_path(region, source, output_root) / "py_importance.png", save)


def plot_probability_map(region, source=DEFAULT_SOURCE, save=False, output_root=None):
    """Mean predicted conflict probability per polygon.

    output_for_REF.gpkg holds min_prob_1, probability_of_conflict and
    max_prob_1. probability_of_conflict is the mean across runs and is the one
    to map; picking the first column whose name contains "prob" would silently
    select the minimum.
    """

    predictions = read_predictions(region, source, output_root)

    if "probability_of_conflict" not in predictions.columns:
        raise ValueError(
            "output_for_REF.gpkg has no `probability_of_conflict` column; "
            f"found {list(predictions.columns)}"
        )

    fig, ax = plt.subplots(figsize=(12, 10))
    predictions.plot(
        column="probability_of_conflict",
        ax=ax,
        cmap="inferno",
        vmin=0,
        vmax=1,
        legend=True,
        legend_kwds={"label": "Mean conflict probability", "shrink": 0.6},
    )
    ax.set_title(f"Conflict probability: {region}")
    ax.set_axis_off()
    plt.tight_layout()
    _finish(get_output_path(region, source, output_root) / "py_probability_map.png", save)


PLOTS = [
    ("Variable importance", plot_importance),
    ("Metrics distribution", plot_metrics_distribution),
    ("Probability map", plot_probability_map),
    ("Selected polygons", plot_polygons),
    ("Selected conflicts", plot_conflicts),
    ("Indicator correlation", plot_correlation),
]


def plot_all(region, source=DEFAULT_SOURCE, save=False, output_root=None):
    """Runs every plot, reporting rather than raising on individual failures."""

    print(f"\n  plots for {region} ({source}):")
    for name, func in PLOTS:
        try:
            print(f"    {name}...")
            func(region, source=source, save=save, output_root=output_root)
        except Exception as e:
            print(f"    {name}: skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Standard plots for a finished CoPro run.",
    )
    ap.add_argument("region", nargs="?", help="region name; omit to list available runs")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="conflict source suffix")
    ap.add_argument("--output-root", required=True, help="CoPro output root")
    ap.add_argument("--save", action="store_true", help="write PNGs instead of showing them")
    args = ap.parse_args()

    if args.region is None:
        print("Available runs:")
        for region, source in list_runs(args.output_root):
            print(f"  {region}  (--source {source})")
        sys.exit(0)

    plot_all(
        args.region,
        source=args.source,
        save=args.save,
        output_root=args.output_root,
    )
