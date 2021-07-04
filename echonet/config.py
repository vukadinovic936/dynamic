"""Sets paths based on configuration files."""

import configparser
import os
import types
import ast

_FILENAME = None
_PARAM = {}
for filename in ["echonet.cfg",
                 ".echonet.cfg",
                 os.path.expanduser("~/echonet.cfg"),
                 os.path.expanduser("~/.echonet.cfg"),
                 ]:
    if os.path.isfile(filename):
        _FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            _PARAM = config["config"]
        break

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATA_DIR=_PARAM.get("data_dir", "/workspace/data/NAS/RV-Milos/RV_data"),
    DATA_DIR_LIST=ast.literal_eval(_PARAM.get("data_dir_list", ["/workspace/data/NAS/RV-Milos/RV_data"])))
