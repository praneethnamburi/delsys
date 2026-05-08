"""ICA-based artifact cleaning for accelerometer data.

Used to identify and remove impact-related components from IMU traces.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA


def _ica_data_preprocess(lf, sensor_locs, time_slice):
    accs = []
    for loc in sensor_locs:
        if time_slice is None:
            acc_raw = lf[loc].acc()
        else:
            acc_raw = lf[loc].acc[time_slice]()
        for i in range(acc_raw.shape[1]):
            acc = acc_raw[:, i]
            acc = acc - np.mean(acc)
            accs.append(acc)
    accs = np.array(accs).T
    return accs


def ica_components(lf, sensor_locs=None, time_slice=None, n_components=None, showplot=True):
    """Run this to identify which components are related to impacts."""
    data = _ica_data_preprocess(lf, sensor_locs, time_slice)
    if n_components is None:
        n_components = data.shape[1]
    transformer = FastICA(random_state=0, n_components=n_components)
    x_transformed = transformer.fit_transform(data)
    matrix = transformer.mixing_
    if showplot:
        fig, axs = plt.subplots(n_components, 1, sharex=True)
        for i in range(n_components):
            axs[i].plot(data[:, i])
            axs[i].plot(x_transformed[:, i])
            axs[i].set_ylabel(i)
        plt.legend(['Raw Data', 'ICA Component'])
        plt.show(block=False)
    return x_transformed, matrix


def ica_cleaning(lf, sensor_locs=None, components_to_remove=None, time_slice=None, n_components=None):
    """Once columns have been identified from ``ica_components``, use this to perform the cleaning."""
    fname_info = lf.fname.split('.')[0] + '_ica_settings.json'
    if os.path.exists(fname_info):
        with open(fname_info) as f:
            ica_settings = json.load(f)
        if components_to_remove is None:
            components_to_remove = ica_settings['components_to_remove']
        if sensor_locs is None:
            sensor_locs = ica_settings['sensor_locs']
        if time_slice is None:
            time_slice = slice(ica_settings['time_slice'][0], ica_settings['time_slice'][1])
        if n_components is None:
            n_components = ica_settings['n_components']
    else:
        if isinstance(time_slice, slice):
            time_slice_json = [time_slice.start, time_slice.stop]
        else:
            time_slice_json = time_slice
        ica_settings = {
            'fname': lf.fname,
            'sensor_locs': sensor_locs,
            'components_to_remove': components_to_remove,
            'time_slice': time_slice_json,
            'n_components': n_components,
        }
        with open(fname_info, 'w') as fw:
            json.dump(ica_settings, fw)
    data = _ica_data_preprocess(lf, sensor_locs, time_slice)
    n_components = data.shape[1]
    S, A = ica_components(lf, sensor_locs, time_slice, n_components, showplot=False)
    for column in components_to_remove:
        S[:, column] = 0
    cleaned_data = S @ A.T
    _, axs = plt.subplots(n_components, 1, sharex=True)
    for i in range(n_components):
        axs[i].plot(data[:, i])
        axs[i].plot(cleaned_data[:, i])
    plt.legend(['raw data', 'cleaned data'])
    plt.show(block=False)
    return cleaned_data
