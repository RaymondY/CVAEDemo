import csv
import os
import pyasn
import random
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset, DataLoader
from config import DefaultConfig

config = DefaultConfig()


class InitAddressDataset(Dataset):
    def __init__(self, prefix_index):
        self.prefix_index = prefix_index
        self.path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
        self.data = np.loadtxt(self.path, delimiter=',').astype(np.int32)
        self.sample_num = self.data.shape[0]
        print(self.data.shape)
        self.labels, self.cluster_num = self.clustering()

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
        # update_init_cluster_info(self.prefix_index, cluster_info)
        with open(config.init_cluster_info_path, 'r') as f:
            lines = f.readlines()
        # join the cluster_labels with ','
        cluster_labels = ','.join([str(i) for i in cluster_info])
        lines[self.prefix_index] = cluster_labels + '\n'
        with open(config.init_cluster_info_path, 'w') as f:
            for line in lines:
                f.write(line)
        cluster_result_one_hot = np.eye(cluster_num)[cluster_result].astype(int)
        return cluster_result_one_hot, cluster_num

    def get_cluster_num(self):
        return self.cluster_num

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class IterAddressDataset(Dataset):
    def __init__(self, prefix_index):
        self.prefix_index = prefix_index
        # read new addresses from config.zmap_result_path + f"no_alias_{prefix}.txt"
        # the first column is the address, the second column is the label
        self.path = config.new_address_path + f"no_alias_{self.prefix_index}.txt"
        self.data_with_label = np.loadtxt(self.path, delimiter=',')
        # turn address into vector by format_str_to_vector function
        self.data = np.array([format_str_to_vector(address) for address in self.data_with_label[:, 0]])
        self.labels = self.data_with_label[:, 1].astype(int)
        self.cluster_num = np.unique(self.labels).shape[0]
        # turn label into one-hot vector
        self.labels = np.eye(self.cluster_num)[self.data_with_label[:, 1].astype(int)].astype(int)

    def get_cluster_num(self):
        return self.cluster_num

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class InitGenerateDataset(Dataset):
    def __init__(self, cluster_distribution, budget):
        self.cluster_distribution = cluster_distribution
        self.cluster_num = len(self.cluster_distribution)
        self.budget = budget
        temp_sum_cluster_labels = np.sum(self.cluster_distribution)
        # budget according to the distribution of cluster labels
        self.budget_per_cluster = np.array(self.cluster_distribution / temp_sum_cluster_labels)
        self.budget_per_cluster = np.round(self.budget_per_cluster * self.budget).astype(int)
        print(self.budget_per_cluster)
        # update the number of samples
        self.budget = np.sum(self.budget_per_cluster)
        # generate cluster labels
        self.cluster_labels = np.zeros(self.budget, dtype=int)
        start_index = 0
        for i in range(self.cluster_num):
            end_index = start_index + self.budget_per_cluster[i]
            self.cluster_labels[start_index:end_index] = i
            start_index = end_index

        self.cluster_labels_one_hot = np.eye(self.cluster_num)[self.cluster_labels].astype(int)
        print(self.cluster_labels_one_hot.shape)

    def get_cluster_num(self):
        return self.cluster_num

    def __getitem__(self, index):
        return self.cluster_labels_one_hot[index]

    def __len__(self):
        return self.budget


class IterGenerateDataset(Dataset):
    def __init__(self, cluster_distribution, budget):
        # the prefix_index refers to model index
        # generate sample_num cluster labels for prefix_index
        self.cluster_distribution = cluster_distribution
        self.cluster_num = len(self.cluster_distribution)
        self.budget = budget
        temp_sum_cluster_labels = np.sum(self.cluster_distribution)
        # sample according to the distribution of cluster labels
        self.sample_per_cluster = np.array(self.cluster_distribution / temp_sum_cluster_labels)
        self.sample_per_cluster = np.round(self.sample_per_cluster * self.budget).astype(int)
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

    def get_cluster_num(self):
        return self.cluster_num

    def __getitem__(self, index):
        return self.cluster_labels_one_hot[index]

    def __len__(self):
        return self.sample_num


def load_init_train_data(prefix_index):
    dataset = InitAddressDataset(prefix_index)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    return dataloader


def load_init_probe_label(cluster_distribution, budget):
    dataset = InitGenerateDataset(cluster_distribution, budget)
    dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=False)
    return dataloader


def load_iter_train_data(prefix_index):
    dataset = IterAddressDataset(prefix_index)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    return dataloader


def load_iter_probe_label(cluster_distribution, budget):
    dataset = IterGenerateDataset(cluster_distribution, budget)
    dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=False)
    return dataloader


def remove_init_duplicate(prefix_index, ipv6_set):
    # check if the ipv6 address is in the training set
    # ipv6_set is a dict, the key is the ipv6 address and the value is the cluster label
    path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
    data = np.loadtxt(path, delimiter=',').astype(np.int32)
    train_ipv6_list = set()
    for i in range(data.shape[0]):
        # format the ipv6 address
        temp_ipv6 = format_vector_to_standard(data[i].tolist())
        train_ipv6_list.add(temp_ipv6)
    # check if the ipv6 address is in the training set, if it is, return remove it
    remove_list = []
    for ipv6 in ipv6_set.keys():
        if ipv6 in train_ipv6_list:
            remove_list.append(ipv6)
    for ipv6 in remove_list:
        ipv6_set.pop(ipv6)


def remove_bank_duplicate(prefix_index, ipv6_set):
    # check if the ipv6 address is in the bank
    # ipv6_set is a dict, the key is the ipv6 address and the value is the cluster label
    # check if the bank prefix_index exists
    path = config.address_bank_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
    if not os.path.exists(path):
        return
    # the bank and keys of ipv6_set are in the standard format
    bank_ipv6_list = set()
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        bank_ipv6_list.add(line.strip())
    # check if the ipv6 address is in the bank, if it is, return remove it
    remove_list = []
    for ipv6 in ipv6_set.keys():
        if ipv6 in bank_ipv6_list:
            remove_list.append(ipv6)
    for ipv6 in remove_list:
        ipv6_set.pop(ipv6)


def format_vector_to_standard(ipv6_list):
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


def format_str_to_vector(ipv6_str):
    # get the standard vector form of an ipv6 address (32 integers)
    # split the address into 8 16-bit segments
    segments = ipv6_str.split(':')
    # if the address ends with "::", append enough zero-filled segments to make 8 total
    if "" in segments:
        empty_index = segments.index("")
        segments[empty_index:empty_index + 1] = ['0'] * (9 - len(segments))
    # convert each segment to an integer and add it to the zmap_result list
    # create a list of 32 integers
    result = []
    for segment in segments:
        # pad each segment with zeros to 4 characters
        segment = segment.zfill(4)
        # convert the segment to 4-bit integers
        result.extend([int(segment[i], 16) for i in range(0, 4)])
    # return np.array(result)
    return result


def format_str_to_standard(ipv6_str):
    ipv6_vector = format_str_to_vector(ipv6_str)
    ipv6_str = format_vector_to_standard(ipv6_vector)
    return ipv6_str


def read_address_without_label(file):
    # the first column is the ipv6 address
    # return a list of ipv6 address
    address_list = []
    with open(file, 'r') as cvsfile:
        reader = csv.reader(cvsfile)
        for row in reader:
            address = row[0]
            address_list.append(address)
    return address_list


def copy_model(prefix_index):
    source_path = config.model_path + "{prefix_index}.pth".format(prefix_index=prefix_index)
    target_path = config.model_fined_path + "{prefix_index}.pth".format(prefix_index=prefix_index)
    shutil.copyfile(source_path, target_path)


############################################################################################################
# need to simplify the code
def read_file(file):
    ip_list = []
    filename = file
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ip = row[0]
            ip_list.append(ip)
    return ip_list

# def tran_ipv6(sim_ip):
#     if sim_ip == "::":
#         return "0000:0000:0000:0000:0000:0000:0000:0000"
#     ip_list = ["0000", "0000", "0000", "0000", "0000", "0000", "0000", "0000"]
#     if sim_ip.startswith("::"):
#         temp_list = sim_ip.split(":")
#         for i in range(0, len(temp_list)):
#             ip_list[i + 8 - len(temp_list)] = ("0000" + temp_list[i])[-4:]
#     elif sim_ip.endswith("::"):
#         temp_list = sim_ip.split(":")
#         for i in range(0, len(temp_list)):
#             ip_list[i] = ("0000" + temp_list[i])[-4:]
#     elif "::" not in sim_ip:
#         temp_list = sim_ip.split(":")
#         for i in range(0, len(temp_list)):
#             ip_list[i] = ("0000" + temp_list[i])[-4:]
#     # elif sim_ip.index("::") > 0:
#     else:
#         temp_list = sim_ip.split("::")
#         temp_list0 = temp_list[0].split(":")
#         # print(temp_list0)
#         for i in range(0, len(temp_list0)):
#             ip_list[i] = ("0000" + temp_list0[i])[-4:]
#         temp_list1 = temp_list[1].split(":")
#         # print(temp_list1)
#         for i in range(0, len(temp_list1)):
#             ip_list[i + 8 - len(temp_list1)] = ("0000" + temp_list1[i])[-4:]
#     # else:
#     #     temp_list = sim_ip.split(":")
#     #     for i in range(0, temp_list):
#     #         ip_list[i] = ("0000" + temp_list[i])[-4:]
#     # print(ip_list)
#     return ":".join(ip_list)


def pandas_entropy(column, base=None):
    a = pd.value_counts(column) / len(column)
    return sum(np.log2(a) * a * (-1))


def alias_detection(df_entropy, prefix_index):
    asndb = pyasn.pyasn('ipasn.20220916.dat')
    as_num = asndb.lookup(df_entropy.iloc[0, 0])[0]
    prefix = asndb.lookup(df_entropy.iloc[0, 0])[1]
    len_prefix = int(int(re.findall('/(.*)', prefix)[0]) / 4)
    df_entropy = df_entropy["address"].str.replace(':', '').astype(str).to_frame()

    list_all = []
    for index, row in df_entropy.iterrows():
        list_temp = list(row["address"])
        list_all.append(list_temp)
    df_entropy = pd.DataFrame(data=list_all)

    entro_list = []
    count_list = []
    for i in range(32):
        per_entro = round(pandas_entropy(df_entropy.iloc[:, i], base=None), 2)
        entro_list.append(per_entro)
        count = df_entropy.iloc[:, i].value_counts()
        # print('************ ', i + 1)
        # print(count)
        count_list.append(count.index[0])
    # print(entro_list)
    # 求相邻差值
    v1 = entro_list[len_prefix - 1:32]
    v2 = entro_list[len_prefix:]
    entro_diff = list(map(lambda x: x[0] - x[1], zip(v2, v1)))
    entro_diff_1 = np.array(list(map(abs, entro_diff)))
    en_values = np.mean(entro_diff_1)
    # print('entro_diff:',list(np.round(np.array(entro_diff),2)),len(entro_diff), en_values)
    for en in entro_diff:
        if en > en_values:
            entro_diff_max = entro_diff.index(en) + len_prefix
            break

    # print(entro_diff_max)
    count_max = df_entropy.iloc[:, entro_diff_max].value_counts()

    adr_list_16 = [count_list] * 16
    df_16 = pd.DataFrame(data=adr_list_16)

    entro_array = np.array(entro_list)
    entro_mean = np.mean(entro_array[len_prefix:])

    df = pd.DataFrame(data=None)
    list_16_seed = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

    test_alias_bit = []
    # for i in range(len_prefix,31):
    #     if entro_array[i] <= entro_mean:
    #         test_alias_bit.append(i)
    #         df_temp = df_16.copy()
    #         for j in range(i+1, 32):
    #             random.shuffle(list_16_seed)
    #             df_temp.iloc[:,j] = list_16_seed
    #             j = j+1
    #         df = pd.concat([df,df_temp], axis=0)

    test_alias_bit.append(entro_diff_max - 1)
    df_temp = df_16.copy()
    for j in range(entro_diff_max, 32):
        random.shuffle(list_16_seed)
        df_temp.iloc[:, j] = list_16_seed
        j = j + 1
    df = pd.concat([df, df_temp], axis=0)

    test_alias_bit_value = []
    for i in range(len(count_max)):
        # entro_diff_max
        value_bit = count_max.index[i]
        test_alias_bit_value.append(value_bit)
        df_temp = df_16.copy()
        df_temp.iloc[:, entro_diff_max] = [value_bit] * 16
        for j in range(entro_diff_max + 1, 32):
            random.shuffle(list_16_seed)
            df_temp.iloc[:, j] = list_16_seed
            j = j + 1
        df = pd.concat([df, df_temp], axis=0)

    df["ipv6"] = df.iloc[:, 0] + df.iloc[:, 1] + df.iloc[:, 2] + df.iloc[:, 3] + ":" + \
                 df.iloc[:, 4] + df.iloc[:, 5] + df.iloc[:, 6] + df.iloc[:, 7] + ":" + \
                 df.iloc[:, 8] + df.iloc[:, 9] + df.iloc[:, 10] + df.iloc[:, 11] + ":" + \
                 df.iloc[:, 12] + df.iloc[:, 13] + df.iloc[:, 14] + df.iloc[:, 15] + ":" + \
                 df.iloc[:, 16] + df.iloc[:, 17] + df.iloc[:, 18] + df.iloc[:, 19] + ":" + \
                 df.iloc[:, 20] + df.iloc[:, 21] + df.iloc[:, 22] + df.iloc[:, 23] + ":" + \
                 df.iloc[:, 24] + df.iloc[:, 25] + df.iloc[:, 26] + df.iloc[:, 27] + ":" + \
                 df.iloc[:, 28] + df.iloc[:, 29] + df.iloc[:, 30] + df.iloc[:, 31]

    df = df[['ipv6']]
    # print(df)
    df_path = config.zmap_result_path + 'alias_det_{prefix_index}.txt'.format(prefix_index=prefix_index)
    df.to_csv(df_path, header=False, index=False)
    zmap_file = 'alias_det_{prefix_index}.txt'.format(prefix_index=prefix_index)

    cmd2 = 'sudo zmap --ipv6-source-ip=2402:f000:6:1401:46a8:42ff:fe43:6d00 --ipv6-target-file=' + df_path + ' -o ' + config.zmap_result_path + 'scan_' + zmap_file + ' -M icmp6_echoscan -B 10M --verbosity=0'
    p = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    hitrate = re.findall(r"\d+\.?\d*", p.communicate()[1][-10:].decode('utf-8'))
    print("alias detection hitrate: ", hitrate)

    if float(hitrate[0]) >= round(16 / df.shape[0], 2):
        list_a = read_file(config.zmap_result_path + 'scan_' + zmap_file)
        ip_list = []
        for i, val in enumerate(list_a):
            # ip_list.append(tran_ipv6(val))
            ip_list.append(format_str_to_standard(val))
        active_alias_df = pd.DataFrame(ip_list, columns=['ipv6'])
        # print(active_alias_df.shape)
        active_alias_df['count'] = [1] * active_alias_df.shape[0]
        df = pd.merge(df, active_alias_df, how='left', on=['ipv6'])
        # print(df)
        j = 0
        # print(len(test_alias_bit), test_alias_bit)
        alias_prefix_str_1 = []
        for i in range(len(test_alias_bit)):
            s1 = df.iloc[j:j + 16, 1].value_counts(dropna=False)
            # print(df.iloc[j:j+16,0:2])
            # print(s1)
            j = j + 16
            if s1.index[0] == 1 and s1[1] == 16:
                alias_bit_low = test_alias_bit[i] + 1
                alias_prefix_list_1 = count_list[0:alias_bit_low]
                insert_fuhao_num = int(alias_bit_low / 4)
                j = 1
                for i in range(1, insert_fuhao_num + 1):
                    j = 4 * i + i - 1
                    alias_prefix_list_1.insert(j, ':')
                alias_prefix_str_1 = ''.join(alias_prefix_list_1)
                break

        # print('**************************')

        all_alias_prefix = []
        insert_fuhao_num = int((entro_diff_max + 1) / 4)
        j = 16 * len(test_alias_bit)
        for i in range(len(test_alias_bit_value)):
            s1 = df.iloc[j:j + 16, 1].value_counts(dropna=False)
            # print(j)
            # print(df.iloc[j:j+16,0:2])
            # print(s1)
            j = j + 16
            if s1.index[0] == 1 and s1[1] == 16:
                # print(count_list[0:entro_diff_max])
                # print('+++++++++++',str(test_alias_bit_value[i]))
                alias_prefix_list = count_list[0:entro_diff_max] + [str(test_alias_bit_value[i])]
                # print(alias_prefix_list)
                n = 1
                for m in range(1, insert_fuhao_num + 1):
                    n = 4 * m + m - 1
                    alias_prefix_list.insert(n, ':')
                alias_prefix_str = ''.join(alias_prefix_list)
                all_alias_prefix.append(alias_prefix_str)
            # else:
            #     print('-----------',str(test_alias_bit_value[i]))
        if alias_prefix_str_1:
            all_alias_prefix = all_alias_prefix + alias_prefix_str_1

        # 删除中间文件
        os.remove(config.zmap_result_path + 'alias_det_{prefix_index}.txt'.format(prefix_index=prefix_index))
        os.remove(config.zmap_result_path + 'scan_alias_det_{prefix_index}.txt'.format(prefix_index=prefix_index))
        return list(set(all_alias_prefix))
    else:
        # 删除中间文件
        os.remove(config.zmap_result_path + 'alias_det_{prefix_index}.txt'.format(prefix_index=prefix_index))
        os.remove(config.zmap_result_path + 'scan_alias_det_{prefix_index}.txt'.format(prefix_index=prefix_index))
        return []
