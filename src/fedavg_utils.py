#!/usr/bin/env python3
import pickle
import base64
import numpy as np
from sklearn.ensemble import IsolationForest


def model_to_bytes(model):
    return base64.b64encode(pickle.dumps(model)).decode('utf-8')


def bytes_to_model(b64_str):
    return pickle.loads(base64.b64decode(b64_str.encode('utf-8')))


def fedavg_aggregate(models, n_samples_list):
    """
    Federated threshold calibration.
    Her client'ın anomali eşiğini (offset_) ağırlıklı ortalama ile birleştirir.
    """
    global_offset = float(np.average(
        [m.offset_ for m in models],
        weights=n_samples_list
    ))
    return global_offset
