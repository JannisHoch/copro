import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
import os
import click


def make_and_collect_output_dirs(
    config: dict, root_dir: click.Path, config_dict: dict
) -> dict:
    """Creates the output folder at location specfied in YAML-file
    and returns dictionary with config-objects and out-dir per run.

    Args:
        config (dict): dictionary containing the parsed configuration-settings of the model.
        root_dir (Path): absolute path to location of configurations-file
        config_dict (dict): dictionary containing config-objects for reference run and all projection.

    Returns:
        dict: dictionary containing config-objects and output directories for reference run and all projection runs.
    """

    # get path to main output directory as specified in cfg-file
    out_dir = os.path.join(root_dir, config["general"]["output_dir"])
    click.echo(f"Saving output to main output folder {out_dir}.")

    # initalize list for all out-dirs
    all_out_dirs = []
    # create reference output dir '_REF' under main output dir
    all_out_dirs.append(os.path.join(out_dir, "_REF"))

    # create reference output dir '_PROJ' under main output dir
    out_dir_proj = os.path.join(out_dir, "_PROJ")
    # create sub-dirs under '_PROJ' for all projection runs
    for key, i in zip(config_dict, range(len(config_dict))):
        # skip the first entry, as this is the reference run which does not need a sub-directory
        if i > 0:
            all_out_dirs.append(os.path.join(out_dir_proj, str(key)))

    # initiate dictionary for config-objects and out-dir per un
    config_outdir_dict = {}
    # for all keys (i.e. run names), assign config-object (i.e. the values) as well as out-dir
    for key, value, i in zip(
        config_dict.keys(), config_dict.values(), range(len(config_dict))
    ):
        config_outdir_dict[key] = [value, all_out_dirs[i]]

    # check if out-dir exists and if not, create it
    for key, value in config_outdir_dict.items():
        # value [0] would be the config-object
        click.echo(f"Creating output-folder {value[1]}.")
        Path.mkdir(Path(value[1]), exist_ok=True, parents=True)

    return config_outdir_dict


def save_to_csv(arg: dict, out_dir: click.Path, fname: str):
    """Saves an dictionary to csv-file.

    Args:
        arg (dict): dictionary or dataframe to be saved.
        out_dir (str): path to output folder.
        fname (str): name of stored item.
    """

    # check if arg is actuall a dict
    if not isinstance(arg, dict):
        raise TypeError("argument is not a dictionary.")
    try:
        arg = pd.DataFrame().from_dict(arg)
    except ValueError:
        arg = pd.DataFrame().from_dict(arg, orient="index")

    # save dataframe as csv
    arg.to_csv(os.path.join(out_dir, fname + ".csv"))


def save_to_npy(arg: Union[dict, pd.DataFrame], out_dir: click.Path, fname: str):
    """Saves an argument (either dictionary or dataframe) to npy-file.

    Args:
        arg (dict or dataframe): dictionary or dataframe to be saved.
        out_dir (str): path to output folder.
        fname (str): name of stored item.
    """

    # if arg is dict, then first create dataframe, then np-array
    if isinstance(arg, dict):
        arg = pd.DataFrame().from_dict(arg)
        arg = arg.to_numpy()
    # if arg is dataframe, directly create np-array
    elif isinstance(arg, pd.DataFrame):
        arg = arg.to_numpy()
    else:
        raise TypeError("argument is not a dictionary or dataframe.")

    # save np-array as npy-file
    np.save(os.path.join(out_dir, fname + ".npy"), arg)
