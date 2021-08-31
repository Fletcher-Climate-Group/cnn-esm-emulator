import pandas as pd
import numpy as np
import pickle
import os.path as osp


def normalize(reference, data):
    if len(reference.shape) == 1:
        reference = np.expand_dims(reference, axis=-1)
        data = np.expand_dims(data, axis=-1)
    for c in range(reference.shape[1]):
        min_c = np.min(reference[:, c])
        max_c = np.max(reference[:, c] - min_c)
        data[:, c] = (data[:, c] - min_c) / max_c
    return np.squeeze(data)


def denormalize(reference, norm_data):
    if len(reference.shape) == 1:
        reference = np.expand_dims(reference, axis=-1)
        norm_data = np.expand_dims(norm_data, axis=-1)
    for c in range(reference.shape[1]):
        min_c = np.min(reference[:, c])
        max_c = np.max(reference[:, c] - min_c)
        norm_data[:, c] = norm_data[:, c] * max_c + min_c
    return np.squeeze(norm_data)


def get_parent_case(sim, sim_type):
    parent_cases = {
        'f45': 'f.e10.F1850.f45_f45.Parent.JGV.2',
        'f19': 'F1850_UQ2019-05-27-13-22_SO4x3_{}.101'.format(sim_type),
        'f09': 'f.e10.F1850.f09_f09.Parent.JGV.2'
    }
    return parent_cases[sim]


def load_data(data_dir='data', sim='f45', mode='sstref'):
    inputs = pd.read_csv(osp.join(data_dir, 'cases01-100_params_headerless.txt'),
                         delim_whitespace=True, header=None).drop(columns=0)
    inputs = inputs[:-1].values
    norm_inputs = normalize(inputs.copy(), inputs.copy())

    sim_data = pickle.load(open(osp.join(data_dir, '{}_average_maps.pkl'.format(sim)), 'rb'))
    sim_cases = sim_data['cases']
    sim_maps = sim_data['maps']
    # lat, lon = sim_data['lat'], sim_data['lon']

    out_idx = [i for i in range(sim_maps.shape[0]) if
               mode in sim_cases[i] and
               'Parent' not in sim_cases[i] and
               sim_cases[i] != get_parent_case(sim, mode)]

    outputs = np.concatenate([np.expand_dims(sim_maps[i], axis=0)
                              for i in out_idx], axis=0)

    if mode == 'sstref':
        parent_idx = sim_cases.index(get_parent_case(sim, mode))
        parent_map = np.expand_dims(sim_maps[parent_idx], axis=0)
        outputs -= parent_map

    elif mode == 'sst2k':
        sstref_idx = [i for i in range(sim_maps.shape[0]) if
                      'sstref' in sim_cases[i] and
                      'Parent' not in sim_cases[i] and
                      sim_cases[i] != get_parent_case(sim, 'sstref')]

        sstref_maps = np.concatenate([np.expand_dims(sim_maps[i], axis=0)
                                      for i in sstref_idx], axis=0)
        outputs -= sstref_maps

    norm_outputs = normalize(outputs.copy(), outputs.copy())
    norm_outputs = norm_outputs.transpose((0, 2, 3, 1))  # channels last
    if sim == 'f45':
        norm_outputs = np.pad(norm_outputs, ((0, 0), (1, 1), (0, 0), (0, 0)))  # pad so width is 48

    if mode == 'sstref':
        return inputs, outputs, norm_inputs, norm_outputs, parent_map
    else:
        return inputs, outputs, norm_inputs, norm_outputs