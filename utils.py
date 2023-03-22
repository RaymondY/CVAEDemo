import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances

import config


class AddressDataset(Dataset):
    def __init__(self, prefix_index, is_train=True):
        self.prefix_index = prefix_index
        self.path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
        self.data = np.loadtxt(self.path, delimiter=',').astype(np.int32)
        self.sample_num = self.data.shape[0]
        print(self.data.shape)
        if is_train:
            self.labels = self.clustering()
            # print the number of each cluster
            print(self.labels.shape)
            print(np.sum(self.labels, axis=0))
        else:
            # self.labels = None
            self.labels = self.clustering()
            # self.data = self.data[0:10, :]

    def clustering(self):
        min_samples = np.round(self.sample_num * config.eps_ratio).astype(int)
        dbscan = DBSCAN(eps=config.eps_threshold, min_samples=min_samples)
        cluster_result = dbscan.fit_predict(self.data)
        # if there are outliers
        if -1 in cluster_result:
            # set all outliers to one cluster
            print('outliers')
            cluster_result = cluster_result + 1
        config.cluster_num = np.unique(cluster_result).shape[0]
        print(config.cluster_num)
        print(np.bincount(cluster_result))
        cluster_result_one_hot = np.eye(config.cluster_num)[cluster_result].astype(int)
        return cluster_result_one_hot

    # def get_normalized_entropy(self):

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


def load_data(prefix_index, is_train=True):
    dataset = AddressDataset(prefix_index, is_train)
    if is_train:
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    return dataloader
