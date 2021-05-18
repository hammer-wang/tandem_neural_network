import pandas as pd
import torch
import torch.utils.data as data

import numpy as np
from sklearn import preprocessing

import pickle as pkl
from torch.utils.data import dataloader


class SiliconColor(data.Dataset):

    def __init__(self, root, split='train'):
        self.root = root
        self.data = pkl.load(open(self.root, 'rb'))

        self.X = np.array([self.data['period'].to_numpy().astype('int'),
                           self.data['height'].to_numpy().astype('int'),
                           self.data['diameter'].to_numpy().astype('int'),
                           self.data['gap'].to_numpy().astype('int')]).T
        self.Y = np.array([self.data['x'].to_numpy().astype('float'),
                           self.data['y'].to_numpy().astype('float'),
                           self.data['Y'].to_numpy().astype('float')]).T
        self.c = self.data['class'].to_numpy().astype('int')
        self.num_classes = len(np.unique(self.c))

        tr_size = int(len(self.X) * 0.6)
        val_size = int(len(self.X) * 0.2)

        self.X_tr, self.Y_tr, self.c_tr = self.X[:
                                                 tr_size], self.Y[:tr_size], self.c[:tr_size]
        self.X_val, self.Y_val, self.c_val = self.X[tr_size:tr_size +
                                                    val_size], self.Y[tr_size:tr_size + val_size], self.c[tr_size:tr_size + val_size]
        self.X_te, self.Y_te, self.c_te = self.X[tr_size +
                                                 val_size:], self.Y[tr_size + val_size:], self.c[tr_size + val_size:]

        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.X_tr)

        if split == 'train':
            self.X, self.Y, self.c = self.scaler.transform(
                self.X_tr), self.Y_tr, self.c_tr
        elif split == 'val':
            self.X, self.Y, self.c = self.scaler.transform(
                self.X_val), self.Y_val, self.c_val
        else:
            self.X, self.Y, self.c = self.scaler.transform(
                self.X_te), self.Y_te, self.c_te

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.c[index]

    def __len__(self):
        return len(self.X)


def get_datasets(root):
    tr_dataset = SiliconColor(root, 'train')
    val_dataset = SiliconColor(root, 'val')
    te_dataset = SiliconColor(root, 'test')

    return tr_dataset, val_dataset, te_dataset


class SiliconColorNShot:

    def __init__(self, root, batch_size, n_way, k_shot, k_query, device=None):
        self.device = device
        self.batch_size = batch_size
        self.n_way = n_way  # this parameter is not in use for regression mode
        self.k_shot = k_shot
        self.k_query = k_query

        self.dt_tr, self.dt_val, self.dt_te = get_datasets(root)
        self.cls2idx_tr = self.class_to_idx(self.dt_tr)
        self.cls2idx_val = self.class_to_idx(self.dt_val)
        self.cls2idx_te = self.class_to_idx(self.dt_te)

    def class_to_idx(self, dt):
        '''
        Build a hash map that maps the class to sample indices
        '''
        cls_to_idx = {}
        for i in range(dt.num_classes):
            cls_to_idx[i] = np.where(dt.c == i)[0]

        return cls_to_idx

    def next(self, mode='train'):
        '''
        first randomly sample tasks, then sample the data points based on the sampled classes
        '''
        if mode == 'train':
            dt = self.dt_tr
            cls2idx = self.cls2idx_tr
        elif mode == 'val':
            dt = self.dt_val
            cls2idx = self.cls2idx_val
        else:
            dt = self.dt_te
            cls2idx = self.cls2idx_te

        # TODO: map the previous classes to new classes.
        # the classes index has to be from 0 to n_way - 1
        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for b in range(self.batch_size):
            selected_classes = np.random.choice(
                np.arange(dt.num_classes), self.n_way)
            le = preprocessing.LabelEncoder()
            le.fit(selected_classes)

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            for cls in selected_classes:
                if len(cls2idx[cls]) >= self.k_shot + self.k_query:
                    sample_indices = np.random.choice(
                        cls2idx[cls], self.k_shot + self.k_query, replace=False)
                else:
                    sample_indices = np.random.choice(
                        cls2idx[cls], self.k_shot + self.k_query, replace=True)

                x_spt.append(dt.X[sample_indices[:self.k_shot]])
                x_qry.append(dt.X[sample_indices[self.k_shot:]])
                y_spt.append(le.transform(dt.c[sample_indices[:self.k_shot]]))
                y_qry.append(le.transform(dt.c[sample_indices[self.k_shot:]]))

            perm = np.random.permutation(self.n_way * self.k_shot)
            x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 4)[perm]
            y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]

            perm = np.random.permutation(self.n_way * self.k_query)
            x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 4)[perm]
            y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        x_spts = np.array(x_spts).astype(np.float32).reshape(
            self.batch_size, self.k_shot * self.n_way, 4)
        y_spts = np.array(y_spts).astype(np.int).reshape(
            self.batch_size, self.k_shot * self.n_way)
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(
            self.batch_size, self.k_query * self.n_way, 4)
        y_qrys = np.array(y_qrys).astype(np.int).reshape(
            self.batch_size, self.k_query * self.n_way)

        x_spts, y_spts, x_qrys, y_qrys = [torch.from_numpy(z).to(
            self.device) for z in [x_spts, y_spts, x_qrys, y_qrys]]

        return x_spts, y_spts, x_qrys, y_qrys


class SiliconColorRegression:

    def __init__(self, root, batch_size, k_shot, k_query, device=None):
        self.device = device
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.k_query = k_query

        self.dt_tr, self.dt_val, self.dt_te = get_datasets(root)

    def next(self, mode='train'):
        '''
        first randomly sample tasks, then sample the data points based on the sampled classes
        '''
        if mode == 'train':
            dt = self.dt_tr
        elif mode == 'val':
            dt = self.dt_val
        else:
            dt = self.dt_te

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for b in range(self.batch_size):

            sample_indicies = np.random.choice(
                len(dt.X), self.k_shot, replace=False)
            x_spt, y_spt = dt.X[sample_indicies], dt.Y[sample_indicies]
            perm = np.random.permutation(self.k_shot)
            x_spt = np.array(x_spt).reshape(self.k_shot, 4)[perm]
            y_spt = np.array(y_spt).reshape(self.k_shot, 3)[perm]

            sample_indicies = np.random.choice(
                len(dt.X), self.k_query, replace=False)
            x_qry, y_qry = dt.X[sample_indicies], dt.Y[sample_indicies]
            perm = np.random.permutation(self.k_query)
            x_qry = np.array(x_qry).reshape(self.k_query, 4)[perm]
            y_qry = np.array(y_qry).reshape(self.k_query, 3)[perm]

            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        x_spts = np.array(x_spts).astype(np.float32).reshape(
            self.batch_size, self.k_shot, 4)
        y_spts = np.array(y_spts).astype(np.float32).reshape(
            self.batch_size, self.k_shot, 3)
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(
            self.batch_size, self.k_query, 4)
        y_qrys = np.array(y_qrys).astype(np.float32).reshape(
            self.batch_size, self.k_query, 3)

        x_spts, y_spts, x_qrys, y_qrys = [torch.from_numpy(z).to(
            self.device) for z in [x_spts, y_spts, x_qrys, y_qrys]]

        return x_spts, y_spts, x_qrys, y_qrys

# Meta Learning INverse design


class SiliconColorTaskLevel(data.Dataset):

    def __init__(self, root, split='train', center_task=None):
        '''
        Args:
            root: the path of the dataset
            split: 'train', 'val', or 'test'
            center_task: if this is provided, then only return neighboring tasks based on the center task, otherwise return the entire split.
        '''
        self.root = root
        self.data = pkl.load(open(self.root, 'rb'))

        self.X_tr, self.Y_tr, self.c_tr = self.get_feature_label(
            self.data['train'])
        self.X_val, self.Y_val, self.c_val = self.get_feature_label(
            self.data['val'])
        self.X_te, self.Y_te, self.c_te = self.get_feature_label(
            self.data['test'])

        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.X_tr)

        if split == 'train':
            self.X, self.Y, self.c = self.scaler.transform(
                self.X_tr), self.Y_tr, self.c_tr
        elif split == 'val':
            self.X, self.Y, self.c = self.scaler.transform(
                self.X_val), self.Y_val, self.c_val
        else:
            self.X, self.Y, self.c = self.scaler.transform(
                self.X_te), self.Y_te, self.c_te

        if center_task:
            c2neighbors = pkl.load(
                open('./data/kmeans_50_adjacency.pkl', 'rb'))
            neighbors = c2neighbors[center_task]
            subset = np.isin(self.c, neighbors)
            self.X, self.Y, self.c = self.X[subset], self.Y[subset], self.c[subset]
            print('Center task {}, neighbors {}, num_neighbors {}'.format(
                center_task, neighbors, len(self.X)))

    @staticmethod
    def get_feature_label(data):
        X = np.array([data['period'].to_numpy().astype('int'),
                      data['height'].to_numpy().astype('int'),
                      data['diameter'].to_numpy().astype('int'),
                      data['gap'].to_numpy().astype('int')]).T

        Y = np.array([data['x'].to_numpy().astype('float'),
                      data['y'].to_numpy().astype('float'),
                      data['Y'].to_numpy().astype('float')]).T

        c = data['class'].to_numpy().astype('int')

        return X, Y, c

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.c[index]

    def __len__(self):
        return len(self.X)


class SiliconColorRegressionTaskSplit:

    '''
    Sample task based on the task split
    '''

    def __init__(self, root, batch_size, k_shot, k_query, device=None, ratio=1):
        self.device = device
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.k_query = k_query

        self.dt = {'train': SiliconColorTaskLevel(root, 'train'),
                   'val': SiliconColorTaskLevel(root, 'val'),
                   'test': SiliconColorTaskLevel(root, 'test')}

        self.split_class = {'train': np.unique(self.dt['train'].c), 'val': np.unique(
            self.dt['val'].c), 'test': np.unique(self.dt['test'].c)}

        self.ratio = ratio
        if ratio < 1:
            num_total = len(self.dt['train'])
            num_train = int(num_total * ratio)
            print(num_train)
            self.dt['train'], _ = torch.utils.data.random_split(
                self.dt['train'], [num_train, num_total - num_train])

        print('training dataset size {}'.format(len(self.dt['train'])))

    def next(self, mode='train', return_task_id=False):
        '''
        first randomly sample tasks, then sample the data points based on the sampled classes
        '''
        dt = self.dt[mode]

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        selected_classes = np.random.choice(
            self.split_class[mode], size=self.batch_size)

        for b in range(self.batch_size):

            if self.ratio == 1:
                sample_indicies = np.random.choice(np.where(dt.c == selected_classes[b])[
                                                   0], self.k_shot + self.k_query, replace=False)
                dataset = dt
            else:
                if len(np.where(dt.dataset.c[dt.indices] == selected_classes[b])[0]) < self.k_shot + self.k_query:
                    sample_indicies = np.random.choice(np.where(dt.dataset.c[dt.indices] == selected_classes[b])[
                                                       0], self.k_shot + self.k_query, replace=True)
                else:
                    sample_indicies = np.random.choice(np.where(dt.dataset.c[dt.indices] == selected_classes[b])[
                                                       0], self.k_shot + self.k_query, replace=False)

                dataset = dt.dataset

            x_spt, y_spt = dataset.X[sample_indicies[:self.k_shot]
                                     ], dataset.Y[sample_indicies[:self.k_shot]]

            perm = np.random.permutation(self.k_shot)
            x_spt = np.array(x_spt).reshape(self.k_shot, 4)[perm]
            y_spt = np.array(y_spt).reshape(self.k_shot, 3)[perm]

            x_qry, y_qry = dataset.X[sample_indicies[self.k_shot:]
                                     ], dataset.Y[sample_indicies[self.k_shot:]]
            perm = np.random.permutation(self.k_query)
            x_qry = np.array(x_qry).reshape(self.k_query, 4)[perm]
            y_qry = np.array(y_qry).reshape(self.k_query, 3)[perm]

            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        x_spts = np.array(x_spts).astype(np.float32).reshape(
            self.batch_size, self.k_shot, 4)
        y_spts = np.array(y_spts).astype(np.float32).reshape(
            self.batch_size, self.k_shot, 3)
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(
            self.batch_size, self.k_query, 4)
        y_qrys = np.array(y_qrys).astype(np.float32).reshape(
            self.batch_size, self.k_query, 3)

        x_spts, y_spts, x_qrys, y_qrys = [torch.from_numpy(z).to(
            self.device) for z in [x_spts, y_spts, x_qrys, y_qrys]]

        if not return_task_id:
            return x_spts, y_spts, x_qrys, y_qrys

        return x_spts, y_spts, x_qrys, y_qrys, selected_classes


if __name__ == '__main__':

    dt = pkl.load(open('./data/si_quad_50.pkl', 'rb'))['test']
    test_classes = np.unique(dt['class'])
    for c in test_classes:
        dataloader = data.DataLoader(SiliconColorTaskLevel(
            './data/si_quad_50.pkl', split='train', center_task=c), batch_size=20)
        for i in range(2):
            x, y, c = next(iter(dataloader))
            print(x.size(), y.size(), c.size())
