import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import pickle as pkl

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import ForwardNet
from dataset import SiliconColorRegressionTaskSplit
from torch.optim.lr_scheduler import StepLR
import tqdm

def train_forward_model(forward_net, args, evaluate=False):
    device = args.device
    datasets = SiliconColorRegressionTaskSplit(
        './data/si_quad_50.pkl', 4, 10, 10, device)
    tr_dataset, val_dataset, te_dataset = datasets.dt['train'], datasets.dt['val'], datasets.dt['test']
    tr_loader = DataLoader(tr_dataset, batch_size=256, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2)
    test_loader = DataLoader(te_dataset, batch_size=128, num_workers=2)

    optimizer = optim.Adam(forward_net.parameters(),
                           lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args.epochs // 5, gamma=0.2)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    val_losses = []
    train_losses = []
    val_loss = val_forward(forward_net, criterion, val_loader, args)
    val_losses.append(val_loss)

    model_path = './trained_models/forward_model_{}_eval{}.pt'.format(
        'MLP', args.eval)
    log_path = './logs/forward_model_{}_eval{}_log.pkl'.format(
        'MLP', args.eval)

    for e in tqdm.tqdm(range(args.epochs)):
        tr_loss = train_forward_epoch(
            forward_net, optimizer, criterion, tr_loader, args)
        val_loss = val_forward(forward_net, criterion, val_loader, args)
        print('Epoch {}, Train loss {:.4f}, Val MSE {:.4f}'.format(
            e, tr_loss, val_loss))

        val_losses.append(val_loss)
        train_losses.append(tr_loss)
        if val_loss < best_val_loss:
            torch.save(forward_net.state_dict(), model_path)
            best_val_loss = val_loss
            print('Serializing model...')

        scheduler.step()

    state_dict = torch.load(model_path)
    forward_net.load_state_dict(state_dict)
    test_loss = val_forward(forward_net, criterion, test_loader, args)
    print('test loss {}'.format(test_loss))

    forward_training_log = {'train_loss': train_losses,
                            'val_mse': val_losses, 'test_mse': test_loss}

    pkl.dump(forward_training_log, open(log_path, 'wb'))


def train_forward_epoch(net, optimizer, criterion, dataloader, args):
    net.train()
    device = args.device

    loss_tr = []
    optimizer.zero_grad()

    for x, y, _ in dataloader:
        x, y = x.float().to(device), y.float().to(device)
        y_pred = net(x)
        loss = criterion(y, y_pred)

        loss.backward()
        optimizer.step()
        loss_tr.append(loss.detach().cpu().item())

    return np.mean(loss_tr)


def val_forward(net, criterion, dataloader, args):
    net.eval()
    device = args.device

    loss_total = []
    for x, y, _ in dataloader:
        x, y = x.float().to(device), y.float().to(device)
        bs = len(x)

        y_pred = net(x)
        loss = criterion(y, y_pred)
        loss_total.append(loss.cpu().detach().item() * bs)

    return np.sum(loss_total) / len(dataloader.dataset.X)


if __name__ == '__main__':

    '''
    Evaluate model: 
    python train_forward_model.py --eval --seed 42
    python train_forward_model.py --seed 10
    '''
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='train a separate model for evaluation')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=int)
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()
    train_eval = args.eval
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    forward_net = ForwardNet().to(args.device)
    train_forward_model(forward_net, args, args.eval)
