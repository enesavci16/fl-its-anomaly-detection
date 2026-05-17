#!/usr/bin/env python3
import json
import numpy as np

def params_to_bytes(offset, n_samples):
    """Sadece offset_ ve n_samples gönder — pickle yok, full model yok."""
    return json.dumps({'offset_': float(offset), 'n_samples': int(n_samples)})

def bytes_to_params(payload_str):
    return json.loads(payload_str)

def fedavg_aggregate(offsets, n_samples_list):
    """Weighted average of client offset_ values."""
    global_offset = float(np.average(offsets, weights=n_samples_list))
    return global_offset
