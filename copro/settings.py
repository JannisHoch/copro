import click
import os
import numpy as np
from shutil import copyfile
from typing import Tuple, Union
from copro import utils, io
from sklearn import ensemble

import yaml


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


def define_model(
    config: dict,
) -> Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]:
    """Defines model based on model configurations.

    Args:
        config (dict): Parsed configuration-settings of the model.

    Returns:
        model: the specified model instance.
    """

    if config["machine_learning"]["model"] in ["Classification", "C"]:
        return ensemble.RandomForestClassifier(random_state=42)
    if config["machine_learning"]["model"] in ["Regression", "R"]:
        return ensemble.RandomForestRegressor(random_state=42)
    raise ValueError(
        "no supported model selected - \
            choose between Classification or Regression"
    )


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


def determine_projection_period(config_REF: dict) -> list:
    """Determines the period for which projections need to be made.
    This is defined as the period between the end year of the reference run
    and the specified projection year for each projection.

    Args:
        config_REF (dict): model configuration-settings for the reference run.

    Returns:
        list: all years of the projection period.
    """

    # get all years of projection period
    projection_period = np.arange(
        config_REF["general"]["y_end"] + 1,
        config_REF["projections"]["proj_2020_to_2023"]["y_end"] + 1,
        1,
    )
    # convert to list
    projection_period = projection_period.tolist()
    click.echo(
        f"The projection period is {projection_period[0]} to {projection_period[-1]}."
    )

    return projection_period
