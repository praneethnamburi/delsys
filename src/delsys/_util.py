"""Small utility helpers used across the delsys package."""
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def decreturn(func):
    """Decorator that lets a function return as dict, np.ndarray, list, or DataFrame.

    The wrapped function must return a dict, and the caller picks the output
    shape via the ``to=`` keyword. Used by EMG and EKG feature-extraction
    methods that produce a dict of named features per window.
    """
    def wrapper(*args, **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Dict, pd.DataFrame, Tuple[List, List]]:
        if kwargs['to'] not in (dict, np.array, np.ndarray, pd.DataFrame, list):
            raise ValueError("Not supported return type. It must be: dict, list, pd.DataFrame, np.array or np.ndarray.")

        ret = func(*args, **kwargs)
        if kwargs['to'] == dict:
            return ret
        elif kwargs['to'] == pd.DataFrame:
            return pd.DataFrame(ret)
        elif kwargs['to'] == np.array or kwargs['to'] == np.ndarray:
            vals = []
            keys = []
            for key, val in ret.items():
                keys.append(key)
                vals.append(val)
            return np.array(vals).transpose(), np.array(keys)
        elif kwargs['to'] == list:
            vals = []
            keys = []
            for key, val in ret.items():
                keys.append(list(key))
                vals.append(list(val))
            return vals, keys
    return wrapper


def _mod_to_attr(name: str) -> str:
    """Convert a modality name (e.g. 'EMGS', 'VO2') to the Sensor attribute name
    that holds it (e.g. 'emg', 'vo2master'). Case-insensitive."""
    if name.upper().startswith('EMG'):
        return 'emg'
    # Link devices use longer attribute names than the modality string
    overrides = {'vo2': 'vo2master', 'hr': 'hrstrap'}
    return overrides.get(name.lower(), name.lower())


def _modset_to_strlist(modset: set) -> list:
    """Return a list of string forms to test a modality against (case-insensitive,
    EMG variants collapsed)."""
    modality_strings = ['EMG' if x.startswith('EMG') else x for x in modset]
    return modality_strings + [x.lower() for x in modality_strings]
