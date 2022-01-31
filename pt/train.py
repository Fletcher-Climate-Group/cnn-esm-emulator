import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add project folder to path

import argparse
from nets import F09Net, Prob_F09Net
from utils.data import load_data
import numpy as np
import os
import os.path as osp
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.plots import plot_predictions, plot_losses
from time import time
import pickle
from utils.losses import probabilistic_loss
import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--exp-dir', default='experiments/pt/single-res')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--prob', action='store_true', help='probabilistic model with mean and variance')
    parser.add_argument('--name', default='')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    exp_dir = osp.join(args.exp_dir, datetime.now().strftime("%Y-%m-%d-%H-%M") if not args.name else args.name)
    args.exp_dir = exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    args.loss = 'probabilistic' if args.prob else args.loss

    with open(osp.join(exp_dir, 'args.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f)

    inputs, outputs, norm_inputs, norm_outputs, parent_maps = load_data(args.data_dir, 'f09', 'sstref')

    shuffle = np.random.permutation(inputs.shape[0])
    test_idx = shuffle[:args.n_test]
    train_idx = shuffle[args.n_test:]

    train_x, train_y = norm_inputs[train_idx], norm_outputs[train_idx]
    test_x, test_y = norm_inputs[test_idx], norm_outputs[test_idx]

    print('train x shape:', train_x.shape)
    print('train y shape:', train_y.shape)
    print('test x shape:', test_x.shape)
    print('test y shape:', test_y.shape)

    # Instantiate models
    if args.prob:
        model = Prob_F09Net(train_x.shape[-1], train_y.shape[-1], args.dropout).to(device)
    else:
        model = F09Net(train_x.shape[-1], train_y.shape[-1], args.dropout).to(device)
    summary(model, (9,))

    train_data = TensorDataset(torch.Tensor(train_x).to(device), torch.Tensor(train_y).to(device))
    train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    spe = int(np.ceil(len(train_data) // args.batch))  # steps per epoch

    test_x = torch.Tensor(test_x).to(device)
    test_y = torch.Tensor(test_y).to(device)

    # Initialize the loss function
    if args.loss == 'mse':
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spe * args.epochs)

    train_loss = []
    val_loss = []
    ts = time()
    for epoch in range(args.epochs):
        print('Epoch {}/{}: lr = {:.9f}'.format(epoch + 1, args.epochs, scheduler.get_lr()[0]))
        model.train()
        running_loss = 0.0
        tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
        counter = 0
        for i, batch_data in enumerate(tk0):
            optimizer.zero_grad()
            (batch_x, batch_y) = batch_data

            with torch.set_grad_enabled(True):
                if args.prob:
                    mu, sigma = model(batch_x)
                    loss = probabilistic_loss(batch_y,
                                              sigma.permute((0, 2, 3, 1)),
                                              mu.permute((0, 2, 3, 1)))
                else:
                    outputs = model(batch_x).permute((0, 2, 3, 1))
                    loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            counter += 1
            tk0.set_postfix(loss=(running_loss / (i + 1)))

        train_loss.append(running_loss / (i + 1))
        model.eval()
        if args.prob:
            epoch_val_loss = probabilistic_loss(batch_y,
                                                sigma.permute((0, 2, 3, 1)),
                                                mu.permute((0, 2, 3, 1))).item()
        else:
            epoch_val_loss = loss_fn(model(test_x).permute((0, 2, 3, 1)), test_y).item()
        val_loss.append(epoch_val_loss)
        print('Validation loss: {:.6f}'.format(val_loss[-1]))

    t = time() - ts
    print('Training took {:d} seconds'.format(int(t)))

    meta = {'train_loss': train_loss,
            'val_loss': val_loss,
            'test_idx': test_idx,
            'training_time': t}

    pickle.dump(meta, open(osp.join(exp_dir, 'meta.pkl'), 'wb'))
    torch.save(model.state_dict(), osp.join(exp_dir, "model.pt"))

    # Predict
    print('Making predictions...')
    if args.predict:
        predict.run(args)