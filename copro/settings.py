import json
import os
from shutil import copyfile
from typing import Tuple, Union

import click
import numpy as np
import yaml
from sklearn import ensemble

from copro import io, utils


def initiate_setup(settings_file: click.Path) -> Tuple[dict, str]:
    """Initiates the model set-up.
    It parses the cfg-file, creates an output folder, copies the cfg-file to the output folder.

    .. example::
        # How the dict should look
        config_dict = {'_REF': [config_REF, out_dir_REF], \
            'run1': [config_run1, out_dir_run1], \
                'run2': [config_run2, out_dir_run2]}

    Args:
        settings_file (Path): path to settings-file (cfg-file).

    Returns:
        dict: dictionary with config-objects and output directories for reference run \
            and all projection runs.
        Path: path to location of the cfg-file for the reference run, serving as main/base directory.
    """

    # print model info, i.e. author names, license info etc.
    utils.print_model_info()

    # get name of directory where cfg-file for reference run is stored
    # all other paths should be relative to this one
    root_dir = os.path.dirname(os.path.abspath(settings_file))

    # parse cfg-file and get config-object for reference run
    config = _parse_settings(settings_file)
    # get dictionary with all config-objects, also for projection runs
    config_dict = _collect_simulation_settings(config, root_dir)

    # get dictionary with all config-objects and all out-dirs
    main_dict = io.make_and_collect_output_dirs(config, root_dir, config_dict)

    # copy cfg-file of reference run to out-dir of reference run
    # for documentation and debugging purposes
    copyfile(
        os.path.abspath(settings_file),
        os.path.join(
            main_dict["_REF"][1], "copy_of_{}".format(os.path.basename(settings_file))
        ),
    )

    return main_dict, root_dir


def _parse_settings(settings_file: click.Path) -> dict:
    """Reads the model configuration YAML-file and returns contant as dictionary.

    Args:
        settings_file (Path): path to settings-file (cfg-file).

    Returns:
        dict: parsed model configuration.
    """

    click.echo(f"Parsing settings from file {settings_file}.")

    with open(settings_file, "r") as stream:
        config = yaml.safe_load(stream)

    return config


def _collect_simulation_settings(config: dict, root_dir: click.Path) -> dict:
    """Collects the configuration settings for the reference run and all projection runs.
    These cfg-files need to be specified one by one in the PROJ_files section of the cfg-file for the reference run.
    The function returns then a dictionary with the name of the run and the associated config-object.

    .. example::
        # How it should look in the cfg-file
        [PROJ_files]
        run1 = cfg_file1
        run2 = cfg_file2

        # How the dict should look
        config_dict = {'_REF': [config_REF], 'run1': [config_run1], 'run2': [config_run2]}

    Args:
        config (dict): dictionariy containing the parsed configuration-settings \
            of the model for the reference run.
        root_dir (Path): path to location of the cfg-file for the reference run.

    Returns:
        dict: dictionary with name and config-object for reference run and all specified projection runs.
    """

    # initiate output dictionary
    config_dict = {}
    # first entry is config-object for reference run
    config_dict["_REF"] = config

    if "projections" in config_dict["_REF"]:

        # loop through all keys and values in PROJ_files section of reference config-object
        for (each_key, each_val) in config_dict["_REF"]["projections"].items():

            # for each value (here representing the cfg-files of the projections), get the absolute path
            each_val = os.path.abspath(
                os.path.join(
                    root_dir, config_dict["_REF"]["projections"][each_key]["file"]
                )
            )

            # parse each config-file specified
            each_config = _parse_settings(each_val)

            # update the output dictionary with key and config-object
            config_dict[each_key] = [each_config]

    return config_dict


# Default hyperparameters of the classifier, used when the config has no
# `machine_learning.hyperparameters` block. Keys in that block replace the
# default for those keys only. The values come from a UCDP-based global
# application (6 subregions) at water-province level; on other data, set the
# block or switch the search on with `machine_learning.tune_hyperparameters`.
DEFAULT_HYPERPARAMETERS = {
    "n_estimators": 200,
    "criterion": "gini",
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "min_impurity_decrease": 0,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": {1: 100},
    "random_state": 42,
}


def define_model(
    config: dict,
) -> Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]:
    """Defines model based on model configurations.

    A classifier carries `DEFAULT_HYPERPARAMETERS`. Keys given in the optional
    `machine_learning.hyperparameters` block of the config replace the default
    for those keys only. Both the evaluation run and the deploy fit get their
    estimator here, so they cannot start from different configurations.

    A regressor is returned with sklearn defaults: the default set is for
    the classifier (class_weight, gini) and says nothing about regression.

    Args:
        config (dict): Parsed configuration-settings of the model.

    Returns:
        model: the specified model instance.
    """

    if config["machine_learning"]["model"] in ["Classification", "C"]:
        hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
        from_config = config["machine_learning"].get("hyperparameters", {})
        hyperparameters.update(from_config)
        # YAML mapping keys can arrive as strings; sklearn matches class_weight
        # keys against classes_, which is integer dtype here.
        if isinstance(hyperparameters["class_weight"], dict):
            hyperparameters["class_weight"] = {
                int(k): v for k, v in hyperparameters["class_weight"].items()
            }
        click.echo("Classifier hyperparameters:")
        for key, value in hyperparameters.items():
            origin = "config" if key in from_config else "package default"
            click.echo(f"...{key}: {value} ({origin})")
        return ensemble.RandomForestClassifier(**hyperparameters)
    if config["machine_learning"]["model"] in ["Regression", "R"]:
        return ensemble.RandomForestRegressor(random_state=42)
    raise ValueError(
        "no supported model selected - \
            choose between Classification or Regression"
    )


def define_tuning(config: dict) -> bool:
    """Whether hyperparameters are searched with GridSearchCV before fitting.

    Read from `machine_learning.tune_hyperparameters`. Off when the config says
    nothing, so a config without the key fits the hyperparameters from
    `define_model` as they are. The evaluation run and the deploy fit both ask
    here, so one cannot search while the other does not.

    Args:
        config (dict): Parsed configuration-settings of the model.

    Returns:
        bool: True if hyperparameters are to be searched.
    """

    tune_hyperparameters = config["machine_learning"].get("tune_hyperparameters", False)
    if tune_hyperparameters:
        click.echo("Hyperparameter search is ON (machine_learning.tune_hyperparameters).")
    else:
        click.echo("Hyperparameter search is OFF - fitting the hyperparameters as defined.")
    return tune_hyperparameters


def describe_fit(config, estimator, tune_hyperparameters, feature_columns) -> dict:
    """Describes what a set of scores was produced with.

    `copro_runner` writes this next to the held-out rows of every evaluation
    run. `scripts/calibrate.py` builds it again from the deploy model's own
    config and compares the two, so a score-to-probability rule cannot be
    fitted on rows from a different predictor set, period or hyperparameter
    configuration than the model it will be applied to.

    Args:
        config (dict): Parsed configuration-settings of the model.
        estimator: the estimator from `define_model`.
        tune_hyperparameters (bool): from `define_tuning`.
        feature_columns (list): feature column names, in model order.

    Returns:
        dict: the description, in the form it has after a JSON round trip.
    """

    description = {
        "feature_set": config["general"].get("feature_set"),
        "y_start": config["general"]["y_start"],
        "y_end": config["general"]["y_end"],
        "scaler": config["machine_learning"]["scaler"],
        "tune_hyperparameters": tune_hyperparameters,
        # as configured, before any search: with the search on the selections
        # differ per run and are not a property of the configuration
        "hyperparameters": {
            key: value
            for key, value in estimator.get_params().items()
            if key in DEFAULT_HYPERPARAMETERS
        },
        "feature_columns": [str(c) for c in feature_columns],
    }
    # round trip, so a description built here compares equal to one read from
    # file (class_weight keys become strings)
    return json.loads(json.dumps(description))


def define_target_var(
    config: dict,
    estimator: Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor],
) -> Union[str, None]:
    """Defines target variable of ML model.
    A target variable needs to be specified for regression models.
    For classification models, it can be provided in the configuration file.
    If not, the default classification approach is used.

    Args:
        config (dict): Parsed configuration-settings of the model.
        estimator (Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]): ML estimator.

    Returns:
        Union[str, None]: Either the target variable or `None`.
    """

    # if target variable is specified, return it
    if "target_var" in config["machine_learning"].keys():
        click.echo(f"Target variable is {config['machine_learning']['target_var']}.")
        return config["machine_learning"]["target_var"]
    # if not, but model is regression, raise error
    if isinstance(estimator, ensemble.RandomForestRegressor):
        raise ValueError("No target variable specified for regression model.")
    # if not, but model is classification, return None and use default classification approach
    click.echo("No target variable specified, using default classification approach.")
    return None


def determine_projection_period(config_REF: dict, projection: str) -> list:
    """Determines the period for which projections need to be made.
    This is defined as the period between the end year of the reference run
    and the y_end of the given projection.

    Args:
        config_REF (dict): model configuration-settings for the reference run.
        projection (str): key of the projection in config_REF["projections"].

    Returns:
        list: all years of the projection period.
    """
    # get all years of projection period
    projection_period = np.arange(
        config_REF["general"]["y_end"] + 1,
        config_REF["projections"][projection]["y_end"] + 1,
        1,
    )
    # convert to list
    projection_period = projection_period.tolist()
    click.echo(
        f"The projection period for {projection} is "
        f"{projection_period[0]} to {projection_period[-1]}."
    )
    return projection_period
