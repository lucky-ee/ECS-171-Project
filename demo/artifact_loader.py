#load saved model on windows (xgboost issue #12459)
# windows cannot load the linux training rng state
# predictions only need the saved model section
# keep the model file and saved settings unchanged
import struct
import sys
from unittest.mock import patch

import joblib
from xgboost import Booster


def portable_model_bytes(buffer):
    # saved snapshot has config and model sections
    # copy the model section exactly as saved
    config_header = b"{L" + struct.pack(">q", 6) + b"Config{"
    model_key = b"L" + struct.pack(">q", 5) + b"Model"
    marker = b"}" + model_key + b"{"
    if not buffer.startswith(config_header) or buffer.count(marker) != 1 or not buffer.endswith(b"}}"):
        raise ValueError("Unexpected XGBoost snapshot format; cannot extract portable Model.")
    start = buffer.index(marker) + 1 + len(model_key)
    return bytearray(buffer[start:-1])


def load_artifact(path):
    #use normal loading on other platforms
    if sys.platform != "win32":
        return joblib.load(path)
    def inference_setstate(booster, state):
        #load model without the training rng state
        Booster.__init__(booster, model_file=portable_model_bytes(state["handle"]))

    # use this method at startup only
    # restore normal loading afterwards
    with patch.object(Booster, "__setstate__", inference_setstate):
        return joblib.load(path)
