#!/usr/bin/env python
"""
Fit the score-to-probability rules for a CoPro DEPLOY model.
(Run once per deploy model, after copro.scripts.deploy and before
copro.scripts.project.)

WHY THIS EXISTS
---------------
The classifier's `y_prob_1` is a score, not a probability. It orders
polygon-years, and the evaluation tests that order, but the level is not
tested by anything and the class weight shifts it on purpose.

This script measures, on held-out rows, how often conflict followed a given
score, and stores that as a rule, so a projection can carry a probability
beside the score. The score itself is never changed: ranking, operating points
and SHAP attribution stay on `y_prob_1`.

WHAT IT DOES
------------
Reads the held-out rows of ONE evaluation run - the per-test-row table
`copro_runner` writes to <output_dir>/eval/<simulation_name>_<strategy>.csv -
and fits one rule per stratum (emerging: no conflict at t-1; ongoing: conflict
at t-1):

  1. rows are sorted by score and cut into consecutive groups of at least
     `--min-rows` rows (default 100), from the top down, rows with the same
     score kept together;
  2. each group gets its observed conflict frequency;
  3. groups that break the rising order are merged (isotonic regression on
     the group table, weighted by rows).

The rule is that table: per group the mean score and the probability. It is
the reliability table made monotone, and nothing else.

WHICH ROWS
----------
The rows file is an explicit argument; this script does not choose a split.
The choice decides what the probabilities mean:

  - leave_polygons_out rows come from models that saw every year, as the
    deploy model does, and cover outcomes of the whole reference period. The
    probabilities are then long-run frequencies.
  - temporal rows cover only the years after the cutoff, from a model that
    never saw them.
  - random rows are close to in-sample and are not a basis for a rule.

`--check-rows` takes the rows of a second run. The rules are applied to them
and expected against observed conflict is written out. With leave_polygons_out
as `--rows` and temporal as `--check-rows`, that table shows how far the
long-run probabilities were off in the most recent years.

THE ROWS MUST BELONG TO THIS MODEL
----------------------------------
A rule is only valid for the configuration whose scores it was fitted on.
`copro_runner` writes <simulation_name>_<strategy>.json beside each rows file,
describing predictor set, period, scaler, hyperparameters and feature columns.
This script builds the same description from the deploy model's own config and
stops if they differ. Rows without a description (written before this script
existed) are refused unless `--allow-unverified-rows` is given; the rule is
then marked unverified in calibration.json.

WRITES
------
    <deploy_dir>/_DEPLOY/calibration.json        the rules, what they rest on
    <deploy_dir>/_DEPLOY/calibration_check.csv   only with --check-rows

`models.run_prediction` applies calibration.json if it is there
(`machine_learning.score_to_probability`). A deploy model without it projects
the score only.

Usage:
    python -m copro.scripts.calibrate <deploy_dir> --rows <eval csv>
    python -m copro.scripts.calibrate <deploy_dir> --rows <eval csv> --check-rows <eval csv>
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import click
import numpy as np
import pandas as pd
from sklearn import isotonic

from copro import machine_learning, settings

# Smallest number of held-out rows a step of a rule may rest on. A frequency
# measured on n rows has a 95% chance margin of at most 0.98 / sqrt(n): about
# +-10 points at 100 rows, +-14 at 50, +-7 at 200. Rows of one polygon in
# neighbouring years are not independent, so the real margin is wider.
DEFAULT_MIN_ROWS_PER_GROUP = 100


def group_rows(score, outcome, min_rows):
    """Cuts rows into consecutive groups along the score, each of at least
    `min_rows` rows, and returns one line per group.

    Rows with the same score are never split: nothing can tell them apart.
    Groups are formed from the HIGHEST score downwards, so the top group is as
    narrow as `min_rows` allows - the top is where the polygons of interest
    sit - and the short remainder ends up at the bottom, where it joins the
    group above it.

    Returns a dataframe with one row per group, lowest scores first, holding
    n, conflicts, score_min, score_max, score_mean and observed.
    """

    table = pd.DataFrame(
        {"score": np.asarray(score, dtype=float), "outcome": np.asarray(outcome, dtype=int)}
    )
    per_score = (
        table.groupby("score")
        .agg(n=("outcome", "size"), conflicts=("outcome", "sum"))
        .sort_index(ascending=False)
        .reset_index()
    )

    group = np.zeros(len(per_score), dtype=int)
    current = 0
    held = 0
    for i, n in enumerate(per_score["n"].to_numpy()):
        group[i] = current
        held += n
        if held >= min_rows:
            current += 1
            held = 0
    if held > 0 and current > 0:
        group[group == current] = current - 1
    per_score["group"] = group
    per_score["score_x_n"] = per_score["score"] * per_score["n"]

    groups = per_score.groupby("group").agg(
        n=("n", "sum"),
        conflicts=("conflicts", "sum"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        score_x_n=("score_x_n", "sum"),
    )
    groups["score_mean"] = groups["score_x_n"] / groups["n"]
    groups["observed"] = groups["conflicts"] / groups["n"]
    groups = groups.drop(columns="score_x_n").sort_values("score_mean").reset_index(drop=True)

    return groups


def read_rows(rows_file, expected_fit, allow_unverified_rows):
    """Reads held-out rows and checks they were produced with `expected_fit`.

    Returns the rows with a `stratum` column, and the description written
    beside them (None if there is none and unverified rows are allowed).
    """

    rows_file = os.path.abspath(rows_file)
    description_file = os.path.splitext(rows_file)[0] + ".json"

    if os.path.isfile(description_file):
        with open(description_file, "r", encoding="utf-8") as f:
            description = json.load(f)
        differing = [
            key for key in expected_fit if description["fit"][key] != expected_fit[key]
        ]
        if differing:
            raise ValueError(
                f"{rows_file} was not produced with the deploy model's configuration.\n"
                + "".join(
                    f"  {key}:\n    rows:   {description['fit'][key]}\n"
                    f"    deploy: {expected_fit[key]}\n"
                    for key in differing
                )
                + "Rerun the evaluation and the deploy fit from one config."
            )
    elif allow_unverified_rows:
        description = None
        click.echo(
            click.style(
                f"WARNING: no {os.path.basename(description_file)} beside the rows - "
                "nothing shows they belong to this deploy model. Continuing because "
                "--allow-unverified-rows was given.",
                fg="red",
            )
        )
    else:
        raise ValueError(
            f"No {description_file}: nothing shows these rows belong to this deploy "
            "model. Rerun the evaluation with this version of copro, or pass "
            "--allow-unverified-rows to fit a rule that is marked unverified."
        )

    click.echo(f"Reading held-out rows from {rows_file}.")
    rows = pd.read_csv(rows_file)
    rows["stratum"] = machine_learning.strata_from_conflict(rows["conflict_t_min_1"])

    return rows, description


def calibrate(
    deploy_dir,
    rows_file,
    check_rows_file=None,
    min_rows=DEFAULT_MIN_ROWS_PER_GROUP,
    allow_unverified_rows=False,
):
    """Fits the rules and writes calibration.json. Returns its content."""

    deploy_dir = os.path.abspath(deploy_dir)
    out_dir_DEPLOY = os.path.join(deploy_dir, "_DEPLOY")

    with open(
        os.path.join(out_dir_DEPLOY, "deploy_manifest.json"), "r", encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    # same config lookup as project.py
    config_matches = sorted(glob.glob(os.path.join(out_dir_DEPLOY, "copy_of_*.yaml")))
    if len(config_matches) != 1:
        raise ValueError(
            f"Expected exactly one reference config copy in {out_dir_DEPLOY}, "
            f"found {len(config_matches)}: {[os.path.basename(f) for f in config_matches]}"
        )
    config = settings._parse_settings(config_matches[0])

    click.echo(click.style("\nINFO: calibration started\n", fg="cyan"))
    click.echo(f"Deploy model: {out_dir_DEPLOY}")
    click.echo(f"Feature set:  {manifest['feature_set']}")
    click.echo(f"Fitted:       {manifest['created_utc']}")

    # the description the rows must carry, built the way copro_runner builds it
    estimator = settings.define_model(config)
    tune_hyperparameters = settings.define_tuning(config)
    expected_fit = settings.describe_fit(
        config, estimator, tune_hyperparameters, manifest["feature_columns"]
    )

    rows, rows_description = read_rows(rows_file, expected_fit, allow_unverified_rows)

    # --- one rule per stratum -----------------------------------------------
    rules = {}
    for stratum in machine_learning.STRATA:
        stratum_rows = rows.loc[rows["stratum"] == stratum]
        groups = group_rows(
            stratum_rows["y_prob_1"], stratum_rows["y_test"], min_rows
        )
        # Merge the groups that break the rising order. Weighted by rows, so a
        # merged step is the observed frequency of the rows it holds.
        groups["probability"] = isotonic.IsotonicRegression(
            y_min=0, y_max=1, increasing=True
        ).fit_transform(groups["score_mean"], groups["observed"], sample_weight=groups["n"])

        rules[stratum] = {
            "n_rows": int(groups["n"].sum()),
            "n_conflicts": int(groups["conflicts"].sum()),
            "n_groups": int(len(groups)),
            "n_steps": int(groups["probability"].nunique()),
        }
        for column in groups.columns:
            rules[stratum][column] = groups[column].tolist()

        click.echo(
            f"\n{stratum}: {rules[stratum]['n_rows']} rows, "
            f"{rules[stratum]['n_conflicts']} with conflict, "
            f"{rules[stratum]['n_groups']} groups, {rules[stratum]['n_steps']} steps."
        )
        click.echo(
            groups[["n", "conflicts", "score_min", "score_max", "observed", "probability"]]
            .round(3)
            .to_string(index=False)
        )

    # --- write --------------------------------------------------------------
    content = {
        "kind": "copro_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # the deploy model these rules were checked against; run_prediction
        # refuses them for a model fitted at any other moment
        "deploy_created_utc": manifest["created_utc"],
        "rows_file": os.path.abspath(rows_file),
        "rows_verified": rows_description is not None,
        "rows_description": rows_description,
        "check_rows_file": None if check_rows_file is None else os.path.abspath(check_rows_file),
        "min_rows_per_group": min_rows,
        "fit": expected_fit,
        "rules": rules,
        "note": (
            "Per stratum: score_mean and probability are the rule. A score is "
            "converted by reading between the two nearest points; below the "
            "first and above the last point the probability is flat. observed "
            "is the conflict frequency of each group before merging."
        ),
    }
    calibration_file = os.path.join(out_dir_DEPLOY, "calibration.json")
    with open(calibration_file, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
        f.write("\n")
    click.echo(f"\nWrote {calibration_file}")

    # --- the rules against a second set of held-out rows ---------------------
    if check_rows_file is not None:
        check_rows, _ = read_rows(check_rows_file, expected_fit, allow_unverified_rows)
        check_rows["calibrated_probability"] = machine_learning.score_to_probability(
            rules, check_rows["stratum"].to_numpy(), check_rows["y_prob_1"].to_numpy()
        )

        check = pd.DataFrame()
        for stratum in machine_learning.STRATA:
            stratum_rows = check_rows.loc[check_rows["stratum"] == stratum]
            groups = group_rows(
                stratum_rows["calibrated_probability"], stratum_rows["y_test"], min_rows
            )
            groups = groups.rename(
                columns={
                    "score_min": "probability_min",
                    "score_max": "probability_max",
                    "score_mean": "probability_mean",
                }
            )
            groups.insert(0, "stratum", stratum)
            check = pd.concat([check, groups], axis=0, ignore_index=True)

            expected = (groups["probability_mean"] * groups["n"]).sum() / groups["n"].sum()
            observed = groups["conflicts"].sum() / groups["n"].sum()
            click.echo(
                f"Check, {stratum}: {groups['n'].sum()} rows, expected conflict "
                f"rate {expected:.3f}, observed {observed:.3f}."
            )

        check_file = os.path.join(out_dir_DEPLOY, "calibration_check.csv")
        check.to_csv(check_file, index=False)
        click.echo(f"Wrote {check_file}")

    click.echo(click.style("\nINFO: calibration finished\n", fg="cyan"))
    return content


def main():
    ap = argparse.ArgumentParser(
        description="Fit the score-to-probability rules for a CoPro deploy model."
    )
    ap.add_argument(
        "deploy_dir", help="the deploy simulation folder holding _DEPLOY"
    )
    ap.add_argument(
        "--rows",
        required=True,
        help="held-out rows to fit the rules on: an eval csv written by "
        "copro_runner, <output_dir>/eval/<simulation_name>_<strategy>.csv",
    )
    ap.add_argument(
        "--check-rows",
        default=None,
        help="held-out rows of a second run to check the rules against. "
        "Writes calibration_check.csv.",
    )
    ap.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS_PER_GROUP,
        help="minimum number of rows per group. Default "
        f"{DEFAULT_MIN_ROWS_PER_GROUP}.",
    )
    ap.add_argument(
        "--allow-unverified-rows",
        action="store_true",
        help="accept rows that carry no description of the run that produced "
        "them. The rules are marked unverified.",
    )
    args = ap.parse_args()

    calibrate(
        deploy_dir=args.deploy_dir,
        rows_file=args.rows,
        check_rows_file=args.check_rows,
        min_rows=args.min_rows,
        allow_unverified_rows=args.allow_unverified_rows,
    )


if __name__ == "__main__":
    main()
