import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from config import DefaultConfig

config = DefaultConfig()


class AddressDataset(Dataset):
    def __init__(self, prefix_index):
        self.prefix_index = prefix_index
        self.path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
        self.data = np.loadtxt(self.path, delimiter=',').astype(np.int32)
        self.sample_num = self.data.shape[0]
        print(self.data.shape)
        self.labels = self.clustering()

    def clustering(self):
        min_samples_data = np.round(self.sample_num * config.eps_ratio).astype(int)
        min_samples = min_samples_data if min_samples_data > config.eps_min_sample else config.eps_min_sample
        # dbscan = DBSCAN(eps=config.eps_threshold, min_samples=min_samples, metric="minkowski", p=0.5)
        dbscan = DBSCAN(eps=config.eps_threshold, min_samples=min_samples, metric="l1")
        cluster_result = dbscan.fit_predict(self.data)
        # if there are outliers
        if -1 in cluster_result:
            # set all outliers to one cluster
            print('outliers exist')
            cluster_result = cluster_result + 1
        cluster_num = np.unique(cluster_result).shape[0]
        cluster_info = np.bincount(cluster_result)
        print(f"cluster_num: {cluster_num}")
        print(f"distribution: {cluster_info}")
        update_cluster_info(self.prefix_index, cluster_info)
        cluster_result_one_hot = np.eye(cluster_num)[cluster_result].astype(int)
        return cluster_result_one_hot

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


class FineTuningAddressDataset(Dataset):
    # refine the address dataset using validated new addresses
    def __init__(self, prefix_index):
        self.prefix_index = prefix_index
        self.dir = config.result_path + '{prefix_index}/'.format(prefix_index=prefix_index)
        self.cluster_num = os.listdir(self.dir).__len__()
        self.data = []
        self.labels = []
        for i in range(self.cluster_num):
            path = self.dir + '{cluster_index}.txt'.format(cluster_index=i)
            # load string data, each row has only one string address
            with open(path, 'r') as f:
                cluster_data = f.readlines()
            cluster_size = cluster_data.__len__()
            if cluster_size == 0:
                print(f"No new address in cluster {i}")
                continue
            # remove the last '\n' in each row
            cluster_data = [address[:-1] for address in cluster_data]
            for j in range(cluster_data.__len__()):
                # convert string address to vector
                cluster_data[j] = get_vector_ipv6(cluster_data[j])
            self.data.append(cluster_data)
            # generate cluster labels one-hot
            cluster_labels = np.zeros(cluster_size, dtype=int)
            cluster_labels[:] = i
            cluster_labels_one_hot = np.eye(self.cluster_num)[cluster_labels].astype(int)
            self.labels.append(cluster_labels_one_hot)
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        print(self.data.shape)
        print(self.labels.shape)
        

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class GenerateDataset(Dataset):
    def __init__(self, prefix_index, sample_num):
        # generate sample_num cluster labels for prefix_index
        self.prefix_index = prefix_index
        self.cluster_distribution = get_cluster_info(prefix_index)
        self.cluster_num = len(self.cluster_distribution)
        self.sample_num = sample_num
        temp_sum_cluster_labels = np.sum(self.cluster_distribution)
        # sample according to the distribution of cluster labels
        self.sample_per_cluster = np.array(self.cluster_distribution / temp_sum_cluster_labels)
        self.sample_per_cluster = np.round(self.sample_per_cluster * self.sample_num).astype(int)
        print(self.sample_per_cluster)
        # update the number of samples
        self.sample_num = np.sum(self.sample_per_cluster)
        # generate cluster labels
        self.cluster_labels = np.zeros(self.sample_num, dtype=int)
        start_index = 0
        for i in range(self.cluster_num):
            end_index = start_index + self.sample_per_cluster[i]
            self.cluster_labels[start_index:end_index] = i
            start_index = end_index

        self.cluster_labels_one_hot = np.eye(self.cluster_num)[self.cluster_labels].astype(int)
        print(self.cluster_labels_one_hot.shape)

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


def load_fine_tuning_data(prefix_index):
    dataset = FineTuningAddressDataset(prefix_index)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    return dataloader


def init_cluster_info():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    with open(config.cluster_info_path, 'w') as f:
        for i in range(prefix_num):
            f.write('-1\n')


def update_cluster_info(prefix_index, cluster_labels):
    # update the cluster info for prefix_index
    # cluster_labels: the number of each cluster
    with open(config.cluster_info_path, 'r') as f:
        lines = f.readlines()
    # join the cluster_labels with ','
    cluster_labels = ','.join([str(i) for i in cluster_labels])
    lines[prefix_index] = cluster_labels + '\n'
    with open(config.cluster_info_path, 'w') as f:
        for line in lines:
            f.write(line)


def get_cluster_info(prefix_index):
    with open(config.cluster_info_path, 'r') as f:
        lines = f.readlines()
    # split the cluster_labels with ','
    cluster_labels = lines[prefix_index].split(',')
    cluster_labels = [int(i) for i in cluster_labels]
    return cluster_labels


def get_cluster_num(prefix_index):
    cluster_labels = get_cluster_info(prefix_index)
    return len(cluster_labels)


def check_duplicate_from_train(prefix_index, ipv6_set):
    # check if the ipv6 address is in the training set
    # ipv6_set is a dict, the key is the ipv6 address and the value is the cluster label
    path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
    data = np.loadtxt(path, delimiter=',').astype(np.int32)
    train_ipv6_list = set()
    for i in range(data.shape[0]):
        # format the ipv6 address
        temp_ipv6 = format_ipv6(data[i].tolist())
        train_ipv6_list.add(temp_ipv6)
    # check if the ipv6 address is in the training set, if it is, return remove it
    remove_list = []
    for ipv6 in ipv6_set.keys():
        if ipv6 in train_ipv6_list:
            remove_list.append(ipv6)
    for ipv6 in remove_list:
        ipv6_set.pop(ipv6)


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


def get_vector_ipv6(ipv6_str):
    # get the standard vector form of an ipv6 address (32 integers)
    # split the address into 8 16-bit segments
    segments = ipv6_str.split(':')
    # if the address ends with "::", append enough zero-filled segments to make 8 total
    if "" in segments:
        empty_index = segments.index("")
        segments[empty_index:empty_index + 1] = ['0'] * (9 - len(segments))
    # convert each segment to an integer and add it to the result list
    # create a list of 32 integers
    result = []
    for segment in segments:
        # pad each segment with zeros to 4 characters
        segment = segment.zfill(4)
        # convert the segment to 4-bit integers
        result.extend([int(segment[i], 16) for i in range(0, 4)])
    return np.array(result)
