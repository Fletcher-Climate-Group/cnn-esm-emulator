import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add project folder to path

import argparse
import os.path as osp
import pickle
import yaml
import numpy as np
import tensorflow as tf
from utils.data import load_data, denormalize
from utils.losses import ss_loss
from utils.plots import plot_predictions

# directory holding model .h5 file and meta data
MODEL_DIR = 'experiments/tf/single-res/base_ss'
PLOT_DIR = osp.join(MODEL_DIR, 'denorm_plots')
RES = 'f09'

args = yaml.safe_load(open(osp.join(MODEL_DIR, 'args.yaml'), 'rb'))  # load experiment arguments as dict
args = argparse.Namespace(**args)  # convert args to Namespace

# args were saved before '--res' option was added to train.py so manually add it
if 'res' not in args:
    args.res = RES

meta = pickle.load(open(osp.join(MODEL_DIR, 'meta.pkl'), 'rb'))  # load meta data

np.random.seed(args.seed)
tf.random.set_seed(args.seed)

inputs, outputs, norm_inputs, norm_outputs, parent_maps = load_data(args.data_dir, args.res, 'sstref')

test_idx = meta['test_idx']  # test sample indices
test_x, test_y = norm_inputs[test_idx], norm_outputs[test_idx]

# load model and make test predictions
model = tf.keras.models.load_model(osp.join(MODEL_DIR, 'model.h5'), compile=False)
print('Loaded model. Making predictions...')
preds = model.predict(test_x)

# compute skill score for sanity check
ssl = ss_loss(np.float32(test_y), preds, reduce_mean=True).numpy()
print('Mean SS across {} test samples and {} output variables:'.format(test_y.shape[0], test_y.shape[-1]))
print('{:.4f}'.format(1 - ssl))

# denormalize predictions
preds = np.transpose(preds, (0, 3, 1, 2))  # channels second
denorm_preds = denormalize(outputs.copy(), preds.copy())  # denormalize using all maps as reference
# denorm_preds += parent_maps  # optionally add parent case maps back to prediction
denorm_preds = np.transpose(denorm_preds, (0, 2, 3, 1))  # channels last for plotting
gt = np.transpose(outputs[test_idx], (0, 2, 3, 1))  # unnormalized gt

plot_predictions(x=denorm_preds[:1],  # second test sample
                 plot_dir=PLOT_DIR,
                 sample_ids=test_idx[:1],
                 gt=gt[:1],
                 out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'],
                 set_bounds='gt',
                 ss_mse_name=True)