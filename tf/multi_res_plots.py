import os.path as osp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml


def result_plot(exp_dir, bl='test', ext='pdf'):
    args = yaml.safe_load(open(osp.join(exp_dir, 'args.yaml'), 'rb'))  # load experiment arguments as dict
    args = argparse.Namespace(**args)  # convert args to Namespace
    SS = 1 - pickle.load(open(osp.join(exp_dir, 'ss.pkl'), 'rb'))
    SS_bl = 1 - pickle.load(open(osp.join(exp_dir, 'ss_bl_{}.pkl'.format(bl)), 'rb'))

    SS = np.mean(SS, axis=2)
    SS_mean = np.mean(SS, axis=-1, keepdims=True)
    SS = np.concatenate((SS_mean, SS), axis=-1)

    SS_bl = np.mean(SS_bl, axis=2)
    SS_bl_mean = np.mean(SS_bl, axis=-1, keepdims=True)
    SS_bl = np.concatenate((SS_bl_mean, SS_bl), axis=-1)

    means = np.mean(SS, axis=1)
    stds = np.std(SS, axis=1)

    means_bl = np.mean(SS_bl, axis=1)
    stds_bl = np.std(SS_bl, axis=1)

    v = ['MEAN', 'AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF']

    fractions = np.array(args.n_hr)

    for i in range(means.shape[-1]):
        plt.figure(figsize=(4, 3))
        plt.xlabel(r'$n_{hr}$')
        if i:
            plt.ylabel(r'1-SS$_k$ ({})'.format(v[i]))
        else:
            plt.ylabel('1-SS')

        # cnn predictions
        l_cnn, = plt.plot(fractions, means[:, i])
        plt.fill_between(fractions, means[:, i] - stds[:, i], means[:, i] + stds[:, i], alpha=0.5)

        # baseline
        if bl == 'train':
            l_bl, = plt.plot(fractions[1:], means_bl[:, i])
            plt.fill_between(fractions[1:], means_bl[:, i] - stds_bl[:, i], means_bl[:, i] + stds_bl[:, i], alpha=0.5)
        else:
            l_bl, = plt.plot(fractions, means_bl[:, i])
            plt.fill_between(fractions, means_bl[:, i] - stds_bl[:, i], means_bl[:, i] + stds_bl[:, i], alpha=0.5)

        plt.ylim([0, 1])
        plt.xlim([0, 80])
        plt.grid()
        if not i:
            plt.legend([l_cnn, l_bl], ['CNN', 'Mean Baseline'])
        plt.tight_layout()

        plt.savefig(osp.join(exp_dir, 'SS_{}_{}.{}'.format(v[i], bl, ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', default='experiments/tf/multi-res/nt_200')
    parser.add_argument('--baseline', default='train')
    parser.add_argument('--ext', default='pdf')
    args = parser.parse_args()

    result_plot(args.exp_dir, args.baseline, args.ext)





