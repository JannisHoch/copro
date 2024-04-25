import pytest
import configparser
import numpy as np
import pandas as pd
from copro import xydata


def create_fake_config():

    config = configparser.ConfigParser()

    config.add_section("general")
    config.set("general", "verbose", str(False))

    return config
