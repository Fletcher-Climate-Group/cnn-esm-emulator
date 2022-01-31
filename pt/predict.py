import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add project folder to path

import argparse
import os.path as osp
import yaml
import torch
from utils.data import load_data
import pickle
from nets import F09Net, Prob_F09Net
from utils.plots import plot_predictions, plot_losses


def run(args):
    with open(osp.join(args.exp_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inputs, outputs, norm_inputs, norm_outputs, parent_maps = load_data(args.data_dir, 'f09', 'sstref')
    test_idx = meta['test_idx']
    test_x, test_y = norm_inputs[test_idx], norm_outputs[test_idx]
    test_x = torch.Tensor(test_x).to(device)
    test_y = torch.Tensor(test_y).to(device)

    if args.prob:
        model = Prob_F09Net(test_x.shape[-1], test_y.shape[-1]).to(device)
    else:
        model = F09Net(test_x.shape[-1], test_y.shape[-1]).to(device)

    model.eval()
    model.load_state_dict(torch.load(osp.join(args.exp_dir, 'model.pt')))

    plot_dir = osp.join(args.exp_dir, 'plots')
    if args.prob:
        mu, sigma = model(test_x)
        mu = mu.permute(0, 2, 3, 1).detach().cpu().numpy()
        sigma = sigma.permute(0, 2, 3, 1).detach().cpu().numpy()
        plot_predictions(x=mu[:1],  # first test sample only
                         plot_dir=plot_dir,
                         sample_ids=test_idx[:1],
                         prefix='mean',
                         out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'])
        plot_predictions(x=sigma[:1],  # first test sample only
                         plot_dir=plot_dir,
                         sample_ids=test_idx[:1],
                         prefix='sigma',
                         out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'],
                         set_bounds=False)
        plot_predictions(x=(test_y[:1].detach().cpu().numpy() - mu[:1]) ** 2,
                         plot_dir=plot_dir,
                         sample_ids=test_idx[:1],
                         prefix='gt-mean',
                         out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'],
                         set_bounds=False)
    else:
        preds = model(test_x).permute((0, 2, 3, 1)).detach().cpu().numpy()
        plot_predictions(x=preds[:1],  # first test sample
                         plot_dir=plot_dir,
                         sample_ids=test_idx[:1],
                         gt=test_y[:1].detach().cpu().numpy(),
                         out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'])

    plot_losses(meta['train_loss'], meta['val_loss'], plot_dir, loss_name=args.loss, start_idx=10, title='PyTorch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--exp-dir', required=True)
    opts = parser.parse_args()

    assert osp.isfile(osp.join(opts.exp_dir, 'model.pt')), 'Model weights not found.'
    with open(osp.join(opts.exp_dir, 'args.yaml'), 'rb') as f:
        args = argparse.Namespace(**yaml.safe_load(f))
    args.exp_dir = opts.exp_dir
    run(args)