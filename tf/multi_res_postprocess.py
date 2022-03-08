import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add project folder to path

import os
import os.path as osp
import pickle
from natsort import natsorted
import numpy as np
from train import ss_loss
import argparse
from utils.data import get_data_lr_hr
import yaml


def process_results(exp_dir):
    files = sorted(os.listdir(exp_dir))
    args = yaml.safe_load(open(osp.join(exp_dir, 'args.yaml'), 'rb'))  # load experiment arguments as dict
    args = argparse.Namespace(**args)  # convert args to Namespace
    meta_files = natsorted([f for f in files if 'meta' in f])
    SS = np.zeros((len(args.n_hr), args.n_trials, args.n_test, 7))
    SS_bl_test = np.zeros((len(args.n_hr), args.n_trials, args.n_test, 7))
    SS_bl_not_test = np.zeros((len(args.n_hr), args.n_trials, args.n_test, 7))
    SS_bl_train = np.zeros((len(args.n_hr) - 1, args.n_trials, args.n_test, 7))
    test_ids = np.zeros((len(args.n_hr), args.n_trials, args.n_test))

    in_lr, out_lr, in_hr, out_hr = get_data_lr_hr(
        'data', args.low_res, args.high_res, args.resize, args.res_ids)

    out_hr = out_hr.transpose((0, 2, 3, 1))

    for m in meta_files:
        meta_data = pickle.load(open(osp.join(exp_dir, m), 'rb'))

        nhr = int(m.split('_')[0][3:])
        idx = args.n_hr.index(nhr)
        trial = int(m.split('_')[1][1:])
        test_ids[idx, trial] = meta_data['hr_test_ids']

        test_mean = np.mean(np.float32(meta_data['test_y']), axis=0, keepdims=True)
        test_mean = np.tile(test_mean, (meta_data['test_y'].shape[0], 1, 1, 1))
        SS_bl_test[idx, trial] = ss_loss(np.float32(meta_data['test_y']), test_mean, reduce_mean=False)

        not_test = out_hr[meta_data['hr_train_ids']]
        not_test_mean = np.mean(not_test, axis=0, keepdims=True)
        not_test_mean = np.tile(not_test_mean, (meta_data['test_y'].shape[0], 1, 1, 1))
        SS_bl_not_test[idx, trial] = ss_loss(np.float32(meta_data['test_y']), np.float32(not_test_mean), reduce_mean=False)

        if nhr > 0:
            train = out_hr[meta_data['hr_train_ids'][:nhr]]
            train_mean = np.mean(train, axis=0, keepdims=True)
            train_mean = np.tile(train_mean, (meta_data['test_y'].shape[0], 1, 1, 1))
            SS_bl_train[idx - 1, trial] = ss_loss(np.float32(meta_data['test_y']), np.float32(train_mean), reduce_mean=False)

        SS[idx, trial] = ss_loss(np.float32(meta_data['test_y']), np.float32(meta_data['preds']), reduce_mean=False)
        print('Finished processing', m)

    pickle.dump(test_ids, open(osp.join(exp_dir, 'test_ids.pkl'), 'wb'))
    pickle.dump(SS, open(osp.join(exp_dir, 'ss.pkl'), 'wb'))
    pickle.dump(SS_bl_test, open(osp.join(exp_dir, 'ss_bl_test.pkl'), 'wb'))
    pickle.dump(SS_bl_not_test, open(osp.join(exp_dir, 'ss_bl_not_test.pkl'), 'wb'))
    pickle.dump(SS_bl_train, open(osp.join(exp_dir, 'ss_bl_train.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', default='experiments/tf/multi-res/nt_200')
    args = parser.parse_args()

    process_results(args.exp_dir)


