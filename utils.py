import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from config import DefaultConfig

config = DefaultConfig()


class AddressDataset(Dataset):
    def __init__(self, prefix_index, is_train=True):
        self.prefix_index = prefix_index
        self.path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
        self.data = np.loadtxt(self.path, delimiter=',').astype(np.int32)
        self.sample_num = self.data.shape[0]
        print(self.data.shape)
        self.labels = self.clustering()
        # if is_train:
        #     self.labels = self.clustering()
        #     # print the number of each cluster
        #     print(self.labels.shape)
        #     print(np.sum(self.labels, axis=0))
        # else:
        #     # self.labels = None
        #     self.labels = self.clustering()

    def clustering(self):
        min_samples = np.round(self.sample_num * config.eps_ratio).astype(int)
        # dbscan = DBSCAN(eps=config.eps_threshold, min_samples=min_samples)
        # cluster_result = dbscan.fit_predict(self.data)
        # if there are outliers
        if -1 in cluster_result:
            # set all outliers to one cluster
            print('outliers exist')
            cluster_result = cluster_result + 1
        cluster_num = np.unique(cluster_result).shape[0]
        print(f"cluster_num: {cluster_num}")
        print(f"distribution: {np.bincount(cluster_result)}")
        update_cluster_num(self.prefix_index, cluster_num)
        cluster_result_one_hot = np.eye(cluster_num)[cluster_result].astype(int)
        return cluster_result_one_hot

    # def get_normalized_entropy(self):

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


class GenerateDataset(Dataset):
    def __init__(self, prefix_index, sample_num):
        # generate sample_num cluster labels for prefix_index
        self.prefix_index = prefix_index
        self.cluster_num = get_cluster_num(prefix_index)
        self.sample_num = sample_num
        self.cluster_labels = np.array([i % self.cluster_num for i in range(self.sample_num)])
        # self.cluster_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        # self.cluster_labels = np.random.randint(0, self.cluster_num, size=self.sample_num)
        # remove the outliers where the cluster_label is 0
        # self.cluster_labels = self.cluster_labels[self.cluster_labels != 0]
        # self.sample_num = self.cluster_labels.shape[0]

        # print(np.bincount(self.cluster_labels))
        self.cluster_labels_one_hot = np.eye(self.cluster_num)[self.cluster_labels].astype(int)
        print(self.cluster_labels_one_hot.shape)
        print(self.cluster_labels_one_hot)

    def __getitem__(self, index):
        return self.cluster_labels_one_hot[index]

    def __len__(self):
        return self.sample_num


def load_train_data(prefix_index):
    dataset = AddressDataset(prefix_index)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    return dataloader


def load_test_data(prefix_index, sample_num=5000):
    dataset = GenerateDataset(prefix_index, sample_num)
    dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=False)
    return dataloader


def init_cluster_num():
    # count .txt files in data_path and init a .txt file
    # to record the number of clusters for each prefix
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    with open(config.cluster_num_path, 'w') as f:
        for i in range(prefix_num):
            f.write('0\n')


def update_cluster_num(prefix_index, cluster_num):
    with open(config.cluster_num_path, 'r') as f:
        lines = f.readlines()
    lines[prefix_index] = str(cluster_num) + '\n'
    with open(config.cluster_num_path, 'w') as f:
        for line in lines:
            f.write(line)


def get_cluster_num(prefix_index):
    with open(config.cluster_num_path, 'r') as f:
        lines = f.readlines()
    return int(lines[prefix_index])


def check_duplicate_from_train(prefix_index, ipv6_set):
    # check if the ipv6 address is in the training set
    path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
    data = np.loadtxt(path, delimiter=',').astype(np.int32)
    train_ipv6_list = set()
    for i in range(data.shape[0]):
        # format the ipv6 address
        temp_ipv6 = format_ipv6(data[i].tolist())
        train_ipv6_list.add(temp_ipv6)
    # check if the ipv6 address is in the training set, if it is, return remove it
    for ipv6 in ipv6_set:
        if ipv6 in train_ipv6_list:
            ipv6_set.remove(ipv6)
    # print the first 10 ipv6 addresses in training set
    print('first 10 ipv6 addresses in training set:')

    return ipv6_set


def format_ipv6(ipv6_list):
    # format an ipv6 address in list form to a standard ipv6 address string
    # ipv6_list: a list of 32 integers
    for i in range(len(ipv6_list)):
        if ipv6_list[i] == 10:
            ipv6_list[i] = 'a'
        elif ipv6_list[i] == 11:
            ipv6_list[i] = 'b'
        elif ipv6_list[i] == 12:
            ipv6_list[i] = 'c'
        elif ipv6_list[i] == 13:
            ipv6_list[i] = 'd'
        elif ipv6_list[i] == 14:
            ipv6_list[i] = 'e'
        elif ipv6_list[i] == 15:
            ipv6_list[i] = 'f'
        else:
            ipv6_list[i] = str(ipv6_list[i])

    ipv6_str_list = []
    for i in range(8):
        temp_section = ipv6_list[i * 4: i * 4 + 4]
        ipv6_str_list.append(''.join([x for x in temp_section]))
    ipv6_str = ':'.join(ipv6_str_list)
    return ipv6_str
