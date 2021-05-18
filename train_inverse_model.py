import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import pickle as pkl

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import ForwardNet, InverseNet, TandemNet
from dataset import SiliconColorRegressionTaskSplit
from utils import load_model, serialize_model

def train_inverse_model(epochs, device):

    print('Start training inverse model...')

    datasets = SiliconColorRegressionTaskSplit(
        './data/si_quad_50.pkl', 4, 10, 10, device)
    tr_dataset, val_dataset, te_dataset = datasets.dt['train'], datasets.dt['val'], datasets.dt['test']
    tr_loader = DataLoader(tr_dataset, batch_size=256, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2)
    test_loader = DataLoader(te_dataset, batch_size=128, num_workers=2)

    # fix the model parameter of the forward model while training the model parameters of the inverse model
    forward_model = ForwardNet()
    forward_model = load_model(forward_model, 'forward_model_MLP_evalFalse.pt').to(device)

    inverse_model = InverseNet().to(device)
    tandem_net = TandemNet(forward_model, inverse_model)
    optimizer = optim.Adam(inverse_model.parameters(),
                           lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for e in range(epochs):
        tr_loss = train_inverse_epoch(
            tandem_net, optimizer, criterion, tr_loader, device)
        val_loss = val_inverse(tandem_net, criterion, val_loader, device)
        print('Epoch {}, Train loss {:.4f}, Val loss {:.4f}'.format(
            e, tr_loss, val_loss))

        if val_loss < best_val_loss:
            serialize_model('./trained_models', tandem_net.inverse_model, 'inverse_model.pt')
            best_val_loss = val_loss
            print('Serializing model...')


def train_inverse_epoch(net, optimizer, criterion, dataloader, device):
    net.train()
    loss_tr = []
    optimizer.zero_grad()

    for x, y, _ in dataloader:
        x, y = x.float().to(device), y.float().to(device)
        # the tandem net takes y (CIE coordinates as the inputs)
        x_pred, y_pred = net(y)
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        loss_tr.append(loss.detach().cpu().item())

    return np.mean(loss_tr)


def val_inverse(net, criterion, dataloader, device):
    net.eval()

    loss_val = []
    for x, y, _ in dataloader:
        x, y = x.float().to(device), y.float().to(device)
        x_pred, y_pred = net(y)
        loss = criterion(y, y_pred)
        loss_val.append(loss.cpu().detach().item())

    return np.mean(loss_val)


if __name__ == '__main__':

    '''
    Evaluate model: 
    python train_inverse_model.py --eval --seed 42
    python train_inverse_model.py --seed 10
    '''
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=int)
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()
    seed = args.seed
    device = args.device
    epochs = args.epochs

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_inverse_model(epochs, device)