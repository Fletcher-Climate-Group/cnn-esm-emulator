import os.path as osp
from hp_search import configs, random_trials
import pickle
import numpy as np

cols = ['exp_name', 'res', 'filters', 'kernel', 'dropout', 'x2_layers', 'params (M)', 'MACs (M)', 'time (s)', 'SS', 'n']
format_row = "{:>18} {:>8}" + "{:>12}" * (len(cols) - 4) + "{:>20} {:>6}"
data = []
for exp_name in configs:
    exp_data = list(configs[exp_name].values())
    ss = []
    mse = []
    n = 0
    for i in range(random_trials):
        try:
            meta = pickle.load(open('experiments/tf/single-res/hp_search/{}_{}/meta.pkl'.format(exp_name, i), 'rb'))
            if i == 0:
                exp_data.extend([
                    '{:0.2f}'.format(meta['parameters'] / 1e6),  # in millions
                    int(meta['operations'] / 1e6),  # in millions
                    int(meta['training_time'])
                ])
            ss.append(1 - meta['val_loss'][-1])
            mse.append(meta['val_mse'][-1])
            n += 1
        except:
            # meta data not found
            if i == 0:
                exp_data.extend(['-', '-', '-'])
    if n:
        exp_data.extend([
            '{:0.3f} +/- {:0.3f}'.format(round(np.mean(ss), 3), round(np.std(ss), 3)),
            # '{:0.3f} +/- {:0.3f}'.format(round(np.mean(mse)*10**4, 3), round(np.std(mse)*10**4, 3)),
        ])
    else:
        exp_data.extend(['-'])
    exp_data.append(n)
    data.append(exp_data)

print(format_row.format(*cols))
for exp_name, row in zip(list(configs.keys()), data):
    print(format_row.format(exp_name, *row))