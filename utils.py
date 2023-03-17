import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances

from config import DefaultConfig

config = DefaultConfig()


class AddressDataset(Dataset):
    def __init__(self, prefix_index, is_train=True):
        self.prefix_index = prefix_index
        self.path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
        self.data = np.loadtxt(self.path, delimiter=',').astype(np.int32)
        if is_train:
            self.labels = self.clustering()
        else:
            self.labels = None
            self.data = self.data[0:10, :]

    def clustering(self):
        hac = AgglomerativeClustering(n_clusters=config.cluster_num, affinity='l1', linkage="complete").fit(self.data)
        # hac = AgglomerativeClustering(n_clusters=config.cluster_num, affinity='hamming').fit(self.data)
        cluster_result = hac.labels_
        print(cluster_result)
        cluster_result_one_hot = np.eye(config.cluster_num)[cluster_result].astype(int)
        return cluster_result_one_hot

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


def load_data(prefix_index, is_train=True):
    dataset = AddressDataset(prefix_index, is_train)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return dataloader


# def load_plane_data(prefix_index, is_train=False):
#     plane_data = np.zeros(shape=(config.sample_num, 32)).astype(int)
#
#     path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
#     plane_data_per_prefix = np.loadtxt(path, delimiter=',').astype(np.int32)
#     plane_data[0:config.sample_num, :] = plane_data_per_prefix
#     # print(plane_data)
#
#     # clustering: hamming distance;
#     # !!! End condition: distance
#     hac = AgglomerativeClustering(n_clusters=config.cluster_num, affinity='hamming')
#     # hac = AgglomerativeClustering(distance_threshold=config.distance_threshold, affinity='hamming')
#     hac.fit(plane_data)
#
#     cluster_result = hac.labels_
#
#     cluster_result_one_hot = np.eye(config.cluster_num)[cluster_result].astype(int)
#
#     # one-hot encoding
#     # plane_data_one_hot = np.zeros(shape=(config.sample_num, config.input_size)).astype(int)
#     # for i in range(config.sample_num):
#     #     plane_data_one_hot[i] = np.eye(17)[plane_data[i]].reshape(-1)
#     #     # print(plane_data_one_hot[i])
#     # # print(plane_data_one_hot.shape)
#     # data = np.hstack((plane_data_one_hot, cluster_result_one_hot)).astype("float32")
#
#     # try binary map encoding
#     # !!!!
#
#     data = np.hstack((plane_data, cluster_result_one_hot)).astype("float32")
#
#     # # print(data)
#     # print(data.shape)
#
#     if is_train:
#         return data
#     else:
#         # 随机选取1000行  进行试探
#         # np.random.shuffle(data)
#         # data_sample = data[:1000, :]
#         # print(data_sample.shape)
#         return tf.convert_to_tensor(data)
