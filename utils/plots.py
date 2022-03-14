import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
import pickle
import argparse
import yaml
from utils.data import get_data_lr_hr
import tensorflow as tf
from utils.losses import ss_loss

warnings.filterwarnings("ignore", category=UserWarning)


def plot_map(x, figsize, save_path, cbar=True, error=False, set_bounds=True):
    assert len(x.shape) == 2, 'can only plot 2d map, len(x.shape) != 2'
    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=180))
    if type(set_bounds) == list or type(set_bounds) == tuple:
        vmin, vmax = set_bounds
    else:
        if set_bounds:
            if error:
                vmin, vmax = -0.2, 0.2
            else:
                vmin, vmax = 0, 1
        else:
            vmin, vmax = None, None
    im = ax.imshow(x,
        transform=ccrs.PlateCarree(central_longitude=180),
        extent=[-180, 180, -90, 90],
        cmap='bwr',
        vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='110m')
    if cbar:
        plt.colorbar(im)
    ax.set_aspect('auto', adjustable=None)
    plt.savefig(save_path)
    print('Saved', save_path)
    plt.close()


def plot_predictions(x, plot_dir, sample_ids=None, gt=None, out_names=None, figsize=(6, 3), ext='png',
                     error=False, prefix='', set_bounds=True, ss_mse_name=False):
    os.makedirs(plot_dir, exist_ok=True)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    if out_names:
        assert x.shape[-1] == len(out_names), 'len(out_names) != x.shape[-1]'
    if error:
        assert gt is None, 'pass error map as x argument with gt=None'
    if not prefix:
        prefix = 'sample' if sample_ids else 'test'
    for i, pred in enumerate(x):
        for j, pred_v in enumerate(pred.transpose((2, 0, 1))):
            test_idx = sample_ids[i] if sample_ids else i
            out_name = out_names[j] if out_names else 'out{}'.format(j)
            if set_bounds == 'gt' and gt is not None:
                vmin, vmax = np.min(gt[i, ..., j]), np.max(gt[i, ..., j])
                set_bounds_gt = (vmin, vmax)
            if gt is not None:
                gt_filename = '{}{}_{}_gt.{}'.format(prefix, test_idx, out_name, ext)
                if set_bounds == 'gt' and gt is not None:
                    plot_map(gt[i, ..., j], figsize, osp.join(plot_dir, gt_filename), set_bounds=set_bounds_gt)
                else:
                    plot_map(gt[i, ..., j], figsize, osp.join(plot_dir, gt_filename), set_bounds=set_bounds)
            pred_filename = '{}{}_{}'.format(prefix, test_idx, out_name)
            pred_filename = pred_filename + '_error' if error else pred_filename
            if ss_mse_name:  # include ss and mse in filename
                ss = 1 - ss_loss(np.float32(gt[i:i+1, ..., j:j+1]), x[i:i+1, ..., j:j+1]).numpy()
                mse = np.mean((gt[i:i+1, ..., j:j+1] - x[i:i+1, ..., j:j+1])**2)
                pred_filename += '_ss{:.3f}_mse{:.3e}'.format(ss, mse)
            pred_filename += '.{}'.format(ext)
            if set_bounds == 'gt' and gt is not None:
                plot_map(x[i, ..., j], figsize, osp.join(plot_dir, pred_filename), error=error, set_bounds=set_bounds_gt)
            else:
                plot_map(x[i, ..., j], figsize, osp.join(plot_dir, pred_filename), error=error, set_bounds=set_bounds)


def plot_multi_res_error(exp_dir, out_names=None, figsize=(6, 3), ext='png'):
    files = os.listdir(exp_dir)

    with open(osp.join(exp_dir, 'args.yaml'), 'rb') as f:
        args = yaml.safe_load(f)

    meta_files = sorted([f for f in files if 'meta' in f
                         and int(f.split('_')[0][3:]) in args['n_hr']])

    meta_data0 = pickle.load(open(osp.join(exp_dir, meta_files[0]), 'rb'))
    error = np.zeros((len(args['n_hr']), args['n_trials'], args['n_test'], *meta_data0['preds'].shape[1:]),
                     dtype=np.float32)

    for m in meta_files:
        nhr = int(m.split('_')[0][3:])
        if nhr in args['n_hr']:
            nhr_idx = args['n_hr'].index(nhr)
            trial = int(m.split('_')[1][1:])
            meta_data = pickle.load(open(osp.join(exp_dir, m), 'rb'))
            error_trial = meta_data['preds'] - meta_data['test_y']
            error[nhr_idx, trial] = error_trial

    mean_error = np.mean(error, axis=(1, 2))
    for i, x in enumerate(mean_error):
        plot_predictions(x,
                         osp.join(exp_dir, 'plots', 'mean_error'),
                         out_names=out_names,
                         figsize=figsize,
                         ext=ext,
                         error=True,
                         prefix='nhr{}'.format(args['n_hr'][i]))


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), loc='upper left', ncol=len(data.keys()))


def plot_feature_importance(exp_dir, nhr_scores, out_names=None, ext='pdf'):
    with open(osp.join(exp_dir, 'args.yaml'), 'rb') as f:
        args = argparse.Namespace(**yaml.safe_load(f))

    in_lr, out_lr, in_hr, out_hr = get_data_lr_hr(
        args.data_dir, args.low_res, args.high_res, args.resize, args.res_ids)

    np.random.seed(0)
    files = sorted(os.listdir(exp_dir))
    meta_files = sorted([f for f in files if 'meta' in f])

    for m in meta_files:
        nhr = int(m.split('_')[0][3:])
        trial = int(m.split('_')[1][1:])
        if nhr != nhr_scores:
            continue

        meta_data = pickle.load(open(osp.join(exp_dir, m), 'rb'))

        x_hr = in_hr[meta_data['hr_train_ids'][:nhr]]
        x = np.concatenate((in_lr, x_hr), axis=0)
        y_hr = out_hr[meta_data['hr_train_ids'][:nhr]]
        y = np.concatenate((out_lr, y_hr), axis=0).transpose([0, 2, 3, 1]).astype(np.float32)
        SS = np.zeros((1, args.n_trials, x.shape[0], y.shape[-1]))
        SS_f = np.zeros((1, args.n_trials, in_lr.shape[-1], x.shape[0], y.shape[-1]))
        idx = 0

        model = tf.keras.models.load_model(osp.join(exp_dir, 'nhr{}_t{}.h5'.format(nhr, trial)), compile=False)
        preds = model(x, training=False)
        SS[idx, trial] = 1 - ss_loss(y, preds, reduce_mean=False)

        for i in range(x.shape[-1]):
            x_shuffled = x.copy()
            x_shuffled[:, i] = x_shuffled[np.random.permutation(x_shuffled.shape[0]), i]
            preds_shuffled = model(x_shuffled, training=False)
            SS_f[idx, trial, i] = 1 - ss_loss(y, preds_shuffled, reduce_mean=False)
        print('Processed', m)

        # generate plots
        SS = np.expand_dims(SS, axis=2)
        SS = SS.transpose((0, 4, 2, 1, 3))

        SS_f = SS_f.transpose((0, 4, 2, 1, 3))

        dSS = SS - SS_f
        dSS = np.mean(dSS, axis=-1)
        dSS_mean = np.mean(dSS, axis=-1)
        dSS_mean /= np.max(dSS_mean)
        dSS_std = np.std(dSS, axis=-1)

        if out_names is None:
            v = ['out{}'.format(i) for i in range(dSS.shape[1])]
        else:
            v = out_names

        x = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'res']

        for nhr in range(dSS.shape[0]):
            data = {}
            for output in range(dSS.shape[1]):
                for input_feature in range(dSS.shape[2]):
                    data[v[output]] = dSS_mean[nhr, output]
            fig, ax = plt.subplots(figsize=(12, 3))
            bar_plot(ax, data, total_width=.8, single_width=.9)
            plt.ylim([0, 1])
            plt.ylabel('Feature Importance (Normalized)')
            plt.xticks(range(10), x)
            plot_dir = osp.join(exp_dir, 'plots', 'feature_importance')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(osp.join(plot_dir, 'feature_importance_nhr{}.{}'.format(nhr_scores, ext)))


def plot_losses(train_loss, val_loss, plot_dir, loss_name='', ext='png', start_idx=0, title=''):
    plt.plot(list(range(1, len(train_loss) + 1))[start_idx:], train_loss[start_idx:])
    plt.plot(list(range(1, len(val_loss) + 1))[start_idx:], val_loss[start_idx:])
    plt.xlabel('Epoch')
    plt.ylabel('{} loss'.format(loss_name))
    plt.title(title)
    plt.legend(['train', 'val'])
    plt.tight_layout()
    plt.savefig(osp.join(plot_dir, 'losses.{}'.format(ext)))
