from re import M
import torch
from torch import nn
import numpy as np


class ForwardNet(nn.Module):

    def __init__(self):
        super(ForwardNet, self).__init__()
        self.linear1 = nn.Linear(4, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(128, 3)

    def forward(self, x):
        h = self.relu1(self.linear1(x))
        h = self.relu2(self.linear2(h))
        h = self.relu3(self.linear2(h))
        o = self.out(h)
        return o


class InverseNet(nn.Module):

    def __init__(self, out_transform=nn.Sigmoid()):
        super(InverseNet, self).__init__()
        self.linear1 = nn.Linear(3, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(128, 4)
        self.out_transform = out_transform

    def forward(self, x):
        h = self.relu1(self.linear1(x))
        h = self.relu2(self.linear2(h))
        h = self.relu3(self.linear2(h))
        o = self.out(h)
        if self.out_transform:
            o = self.out_transform(o)
        return o


class TandemNet(nn.Module):

    def __init__(self, forward_model, inverse_model):
        super(TandemNet, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(self, y):
        '''
        Args:
            y: true CIE coordinates
        
        Returns:
            x_: predicted structural parameters
            y_: predicted CIE coordinates for the inversely-designed structure

        '''
        x_ = self.inverse_model(y)
        y_ = self.forward_model(x_)
        return x_, y_


if __name__ == '__main__':

    forward_model = ForwardNet()
    inverse_model = InverseNet()
    tandem_net = TandemNet(forward_model, inverse_model)

    x = torch.rand(128, 3)
    print(forward_model(inverse_model(x)))

    print(tandem_net(x))
