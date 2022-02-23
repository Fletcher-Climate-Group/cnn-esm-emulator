import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add project folder to path

import os, os.path as osp
import pickle
import numpy as np
import tensorflow as tf
from utils.data import load_data
from utils.plots import plot_predictions
import sys

model_root = 'experiments/tf/single-res/hp_search/'
model_dirs = os.listdir(model_root)
model_dirs = [m for m in model_dirs if 'combo_' in m]
ss, mse = [], []
for m in sorted(model_dirs):
    meta = pickle.load(open(osp.join(model_root, m, 'meta.pkl'), 'rb'))
    ss.append(meta['val_loss'][-1])
    mse.append(meta['val_mse'][-1])
    print(m, ss[-1], mse[-1])
seed = np.argmin(ss)
best_model = 'combo_{}'.format(seed)

np.random.seed(seed)
tf.random.set_seed(seed)
inputs, outputs, norm_inputs, norm_outputs, parent_maps = load_data('data', 'f09', 'sstref')
meta = pickle.load(open(osp.join(model_root, best_model, 'meta.pkl'), 'rb'))
test_idx = meta['test_idx']
test_x, test_y = norm_inputs[test_idx], norm_outputs[test_idx]

model = tf.keras.models.load_model(osp.join(model_root, best_model, 'model.h5'), compile=False)

# override trainable parameter for batch norm so it runs in inference mode when using training=True
# source: https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/layers/normalization/batch_normalization.py#L739-L742
for layer in model.layers:
    if 'batch_normalization' in layer.name:
        layer.trainable = False

print('Making plots...')
for i in range(50):
    # run with training=True to turn on dropout
    preds = np.expand_dims(model(test_x, training=True).numpy(), axis=0)
    if i == 0:
        preds_all = preds
    else:
        preds_all = np.concatenate((preds_all, preds), axis=0)
preds_mean = np.mean(preds_all, axis=0)
preds_std = np.std(preds_all, axis=0)

plot_dir = osp.join(model_root, best_model, 'plots')
plot_predictions(x=preds_mean[1:2],  # second test sample
                 plot_dir=plot_dir,
                 sample_ids=test_idx[1:2],
                 gt=test_y[1:2],
                 out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'],
                 prefix='mean_')

plot_predictions(x=preds_std[1:2],  # second test sample
                 plot_dir=plot_dir,
                 sample_ids=test_idx[1:2],
                 gt=None,
                 out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'],
                 prefix='std_',
                 set_bounds=False)
