import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def plot_map(x, figsize, save_path, cbar=True):
    assert len(x.shape) == 2, 'can only plot 2d map, len(x.shape) != 2'
    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=180))
    im = ax.imshow(x,
        transform=ccrs.PlateCarree(central_longitude=180),
        extent=[-180, 180, -90, 90],
        cmap='bwr',
        vmin=0, vmax=1)
    ax.coastlines(resolution='110m')
    if cbar:
        plt.colorbar(im)
    ax.set_aspect('auto', adjustable=None)
    plt.savefig(save_path)
    print('Saved', save_path)
    plt.close()


def plot_predictions(x, exp_dir, sample_ids=None, gt=None, out_names=None, figsize=(6, 3), ext='png'):
    plot_dir = osp.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    if out_names:
        assert x.shape[-1] == len(out_names), 'len(out_names) != x.shape[-1]'
    prefix = 'sample' if sample_ids else 'test'
    for i, pred in enumerate(x):
        for j, pred_v in enumerate(pred.transpose((2, 0, 1))):
            test_idx = sample_ids[i] if sample_ids else i
            out_name = out_names[j] if out_names else 'out{}'.format(j)
            if gt is not None:
                gt_filename = '{}{}_{}_gt.{}'.format(prefix, test_idx, out_name, ext)
                plot_map(gt[i, ..., j], figsize, osp.join(plot_dir, gt_filename))
            pred_filename = '{}{}_{}.{}'.format(prefix, test_idx, out_name, ext)
            plot_map(x[i, ..., j], figsize, osp.join(plot_dir, pred_filename))
