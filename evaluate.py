import os
import re
import subprocess
import pandas as pd
import torch
from config import DefaultConfig
from cvae import CVAE
from utils import *

config = DefaultConfig()
device = config.device


def probe(model, test_loader):
    # return new address with cluster label together
    generated_address_with_label = dict()
    model.eval()
    with torch.no_grad():
        for batch, c in enumerate(test_loader):
            c = c.float().to(device)
            # generate
            pred_x = model.generate(c).squeeze(1)
            # round pred_x to int and limit to [0, 15]
            pred_x[pred_x > 15] = 15
            pred_x[pred_x < 0] = 0
            # pred_x = pred_x.round().int()
            pred_x = pred_x.round().int()
            pred_x = pred_x.cpu().numpy()
            # save to new_address
            for i in range(pred_x.shape[0]):
                temp_address = format_vector_to_standard(pred_x[i].tolist())
                # turn one-hot to label
                temp_label = c[i].argmax().item()
                if temp_address not in generated_address_with_label:
                    generated_address_with_label[temp_address] = temp_label
                elif generated_address_with_label[temp_address] != temp_label:
                    print(f"Error: {temp_address} has different cluster label.")
                    print(f"Old label: {generated_address_with_label[temp_address]}, new label: {temp_label}")
                    generated_address_with_label[temp_address] = temp_label
    return generated_address_with_label


def init_probe_specific_prefix(budget):
    prefix_list = os.listdir(config.iter_model_path)
    prefix_list = [int(re.sub(r"\.pth$", "", prefix)) for prefix in prefix_list]
    prefix_cluster_distribution_dict = dict()
    prefix_cluster_sum_dict = dict()
    cluster_sum = 0
    with open(config.init_cluster_info_path, 'r') as f:
        lines = f.readlines()
        # the prefix is the index of the line
        for prefix in prefix_list:
            cluster_distribution = lines[prefix].strip().split(",")
            prefix_cluster_distribution_dict[prefix] = [int(i) for i in cluster_distribution.split(",")]
            # get the sum of cluster distribution
            temp_sum = sum(prefix_cluster_distribution_dict[prefix])
            prefix_cluster_sum_dict[prefix] = temp_sum
            cluster_sum += temp_sum

    # save new cluster distribution for each prefix
    new_prefix_cluster_distribution_dict = dict()
    new_prefix_hit_rate_dict = dict()
    new_prefix_hit_rate_no_alias_dict = dict()
    new_list_alias_prefix_dict = dict()
    used_budget = 0

    for prefix in prefix_cluster_distribution_dict.keys():
        cluster_distribution = prefix_cluster_distribution_dict[prefix]
        cluster_num = len(cluster_distribution)
        temp_sum = prefix_cluster_sum_dict[prefix]
        # round
        temp_budget = round(budget * temp_sum / cluster_sum)
        probe_loader = load_init_probe_label(cluster_distribution, temp_budget)
        # load corresponding model
        model = CVAE(cluster_num).to(device)
        model.load_state_dict(torch.load(config.init_model_path + f"{prefix}.pth"))
        generated_address_with_label = probe(model, probe_loader)
        # remove duplicate
        remove_init_duplicate(prefix, generated_address_with_label)
        remove_bank_duplicate(prefix, generated_address_with_label)
        generated_address_num = len(generated_address_with_label)
        # create config.address_bank_path/prefix.txt and write into it
        with open(config.address_bank_path + f"{prefix}.txt", 'a') as f:
            for address in generated_address_with_label.keys():
                f.write(address + "\n")
        # create config.generated_address_path/prefix.txt and write into it
        with open(config.generated_address_path + f"{prefix}.txt", 'w') as f:
            for address in generated_address_with_label.keys():
                f.write(address + "\n")

        generated_address_path = config.generated_address_path + '{prefix}'.format(prefix=prefix) + ".txt"
        zmap_result_path = config.zmap_result_path + '{prefix}'.format(prefix=prefix) + ".txt"
        print(f"Running zmap for prefix {prefix}...")
        cmd = (f"sudo zmap --ipv6-source-ip={config} "
               f"--ipv6-target-file={generated_address_path} "
               f"-o {zmap_result_path} -M icmp6_echoscan -B 10M --verbosity=0")

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        hit_rate = re.findall(r"\d+\.?\d*", p.communicate()[1][-10:].decode('utf-8'))
        print(f"Hit rate: {hit_rate}%")

        # read zmap result and format from string to standard address
        zmap_active_address = []
        with open(zmap_result_path, 'r') as f:
            for line in f:
                zmap_active_address.append(format_str_to_standard(line.strip()))
        # merge zmap_active_address and generated_address_with_label
        # in this way, we can get the cluster label for each active address
        active_address_with_label = dict()
        for address in zmap_active_address:
            if address in generated_address_with_label:
                active_address_with_label[address] = generated_address_with_label[address]

        # generate df columns=['address', 'label']
        df_active_address_label = pd.DataFrame(active_address_with_label.items(), columns=['address', 'label'])
        active_address_num = df_active_address_label.shape[0]
        # remove alias prefix
        list_alias_prefix = []
        if df_active_address_label.shape[0] > 1:
            list_alias_prefix = list_alias_prefix + alias_detection(df_active_address_label, prefix)
            print('alias prefix :', list_alias_prefix)
            if list_alias_prefix:
                # 删除别名地址影响
                for alias_prefix in list_alias_prefix:
                    df_active_address_label = df_active_address_label.loc[
                        df_active_address_label['address'].apply(
                            lambda s: re.search(alias_prefix, s) is None)].reset_index(
                        drop=True)
                print('active address after del alias:', df_active_address_label.shape[0])

        no_alias_active_address_num = df_active_address_label.shape[0]
        alias_active_address_num = active_address_num - no_alias_active_address_num
        hit_rate_no_alias = round(no_alias_active_address_num /
                                  (generated_address_num - alias_active_address_num) * 100, 2)
        print('hit_rate_no_alias: ', hit_rate_no_alias)

        # save the active address with label into config.new_address_path/prefix.txt
        df_active_address_label.to_csv(config.new_address_path + f"{prefix}.txt",
                                       sep=',', header=False, index=False)
        # add the active address with label into config.new_address_path/all_prefix.txt
        with open(config.new_address_path + f"all_{prefix}.txt", 'a') as f:
            for i in range(df_active_address_label.shape[0]):
                f.write(df_active_address_label.iloc[i, 0] + "," + str(df_active_address_label.iloc[i, 1]) + "\n")

        # count the number of active address for each label in a list
        new_cluster_distribution = [0] * cluster_num
        for i in range(cluster_num):
            new_cluster_distribution[i] = df_active_address_label.loc[df_active_address_label['label'] == i].shape[0]
        print(f"Active address number for each label: {new_cluster_distribution}")
        # update the cluster distribution
        new_prefix_cluster_distribution_dict[prefix] = new_cluster_distribution
        # update the hit rate
        new_prefix_hit_rate_dict[prefix] = hit_rate_no_alias
        new_prefix_hit_rate_no_alias_dict[prefix] = hit_rate_no_alias
        new_list_alias_prefix_dict[prefix] = list_alias_prefix
        used_budget += generated_address_num

        # overwrite config.cluster_info_path
        # format: prefix, cluster_distribution\n prefix, cluster_distribution\n
        # example: 1: 0, 99, 3 \n 5: 14, 20 \n
    with open(config.cluster_info_path, 'w') as f:
        for prefix, cluster_distribution in new_prefix_cluster_distribution_dict.items():
            # convert list to string
            cluster_distribution = ','.join(str(i) for i in cluster_distribution)
            f.write(f"{prefix}:{cluster_distribution}\n")


def iter_probe_specific_prefix(budget):
    # the cluster distribution for each prefix is a list
    prefix_cluster_distribution_dict = dict()
    prefix_cluster_sum_dict = dict()
    cluster_sum = 0
    with open(config.cluster_info_path, 'r') as f:
        for line in f:
            prefix, cluster_distribution = line.strip().split(":")
            prefix = int(prefix)
            prefix_cluster_distribution_dict[prefix] = [int(i) for i in cluster_distribution.split(",")]
            # get the sum of cluster distribution
            temp_sum = sum(prefix_cluster_distribution_dict[prefix])
            prefix_cluster_sum_dict[prefix] = temp_sum
            cluster_sum += temp_sum

    # save new cluster distribution for each prefix
    new_prefix_cluster_distribution_dict = dict()
    new_prefix_hit_rate_dict = dict()
    new_prefix_hit_rate_no_alias_dict = dict()
    new_list_alias_prefix_dict = dict()
    used_budget = 0

    for prefix in prefix_cluster_distribution_dict.keys():
        cluster_distribution = prefix_cluster_distribution_dict[prefix]
        cluster_num = len(cluster_distribution)
        temp_sum = prefix_cluster_sum_dict[prefix]
        # round
        temp_budget = round(budget * temp_sum / cluster_sum)
        probe_loader = load_iter_probe_label(cluster_distribution, temp_budget)
        # load corresponding model
        model = CVAE(cluster_num).to(device)
        model.load_state_dict(torch.load(config.iter_model_path + f"{prefix}.pth"))
        generated_address_with_label = probe(model, probe_loader)
        # remove duplicate
        remove_init_duplicate(prefix, generated_address_with_label)
        remove_bank_duplicate(prefix, generated_address_with_label)
        generated_address_num = len(generated_address_with_label)
        # add to config.address_bank_path/prefix.txt
        with open(config.address_bank_path + f"{prefix}.txt", 'a') as f:
            for address in generated_address_with_label.keys():
                f.write(address + "\n")
        # overwrite into config.generated_address_path/prefix.txt
        with open(config.generated_address_path + f"{prefix}.txt", 'w') as f:
            for address in generated_address_with_label.keys():
                f.write(address + "\n")

        generated_address_path = config.generated_address_path + '{prefix}'.format(prefix=prefix) + ".txt"
        zmap_result_path = config.zmap_result_path + '{prefix}'.format(prefix=prefix) + ".txt"
        print(f"Running zmap for prefix {prefix}...")
        cmd = (f"sudo zmap --ipv6-source-ip={config} "
               f"--ipv6-target-file={generated_address_path} "
               f"-o {zmap_result_path} -M icmp6_echoscan -B 10M --verbosity=0")

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        hit_rate = re.findall(r"\d+\.?\d*", p.communicate()[1][-10:].decode('utf-8'))
        print(f"Hit rate: {hit_rate}%")

        # read zmap result and format from string to standard address
        zmap_active_address = []
        with open(zmap_result_path, 'r') as f:
            for line in f:
                zmap_active_address.append(format_str_to_standard(line.strip()))
        # merge zmap_active_address and generated_address_with_label
        # in this way, we can get the cluster label for each active address
        active_address_with_label = dict()
        for address in zmap_active_address:
            if address in generated_address_with_label:
                active_address_with_label[address] = generated_address_with_label[address]

        # generate df columns=['address', 'label']
        df_active_address_label = pd.DataFrame(active_address_with_label.items(), columns=['address', 'label'])
        active_address_num = df_active_address_label.shape[0]
        # remove alias prefix
        list_alias_prefix = []
        if df_active_address_label.shape[0] > 1:
            list_alias_prefix = list_alias_prefix + alias_detection(df_active_address_label, prefix)
            print('alias prefix :', list_alias_prefix)
            if list_alias_prefix:
                # 删除别名地址影响
                for alias_prefix in list_alias_prefix:
                    df_active_address_label = df_active_address_label.loc[
                        df_active_address_label['address'].apply(
                            lambda s: re.search(alias_prefix, s) is None)].reset_index(
                        drop=True)
                print('active address after del alias:', df_active_address_label.shape[0])

        no_alias_active_address_num = df_active_address_label.shape[0]
        alias_active_address_num = active_address_num - no_alias_active_address_num
        hit_rate_no_alias = round(no_alias_active_address_num /
                                  (generated_address_num - alias_active_address_num) * 100, 2)
        print('hit_rate_no_alias: ', hit_rate_no_alias)

        # save the active address with label into config.new_address_path/prefix.txt
        df_active_address_label.to_csv(config.new_address_path + f"{prefix}.txt",
                                       sep=',', header=False, index=False)
        # add the active address with label into config.new_address_path/all_prefix.txt
        with open(config.new_address_path + f"all_{prefix}.txt", 'a') as f:
            for i in range(df_active_address_label.shape[0]):
                f.write(df_active_address_label.iloc[i, 0] + "," + str(df_active_address_label.iloc[i, 1]) + "\n")

        # count the number of active address for each label in a list
        new_cluster_distribution = [0] * cluster_num
        for i in range(cluster_num):
            new_cluster_distribution[i] = df_active_address_label.loc[df_active_address_label['label'] == i].shape[0]
        print(f"Active address number for each label: {new_cluster_distribution}")
        # update the cluster distribution
        new_prefix_cluster_distribution_dict[prefix] = new_cluster_distribution
        # update the hit rate
        new_prefix_hit_rate_dict[prefix] = hit_rate_no_alias
        new_prefix_hit_rate_no_alias_dict[prefix] = hit_rate_no_alias
        new_list_alias_prefix_dict[prefix] = list_alias_prefix
        used_budget += generated_address_num

    # overwrite config.cluster_info_path
    # format: prefix, cluster_distribution\n prefix, cluster_distribution\n
    # example: 1: 0, 99, 3 \n 5: 14, 20 \n
    with open(config.cluster_info_path, 'w') as f:
        for prefix, cluster_distribution in new_prefix_cluster_distribution_dict.items():
            # convert list to string
            cluster_distribution = ','.join(str(i) for i in cluster_distribution)
            f.write(f"{prefix}:{cluster_distribution}\n")

#
# def test_specific_model(prefix, sample_num=5000):
#     print(f"Testing model for prefix {prefix}...")
#     test_loader = load_test_data(prefix, sample_num)
#     cluster_num = get_cluster_num(prefix)
#     model = CVAE(cluster_num).to(device)
#     # load corresponding model
#     model.load_state_dict(torch.load(config.model_path + f"{prefix}.pth"))
#     # the new address saved in a dict
#     new_address_with_labels = test(model, test_loader)
#     print(f"New address number before removing duplicate from train data: {len(new_address_with_labels)}")
#     # remove duplicate from train data
#     check_duplicate_from_train(prefix, new_address_with_labels)
#     print(f"New address number before removing duplicate from address bank: {len(new_address_with_labels)}")
#     # remove duplicate from address bank
#     check_duplicate_from_bank(prefix, new_address_with_labels)
#     print(f"New address number: {len(new_address_with_labels)}")
#
#     # mkdir "prefix" if not exists
#     # if not os.path.exists(config.new_address_path + f"{prefix}"):    os.path.exists(config.zmap_result_path
#     #     os.mkdir(config.new_address_path + f"{prefix}")
#     # else:
#     #     # remove all files in the folder, in case of mixing up old and new files
#     #     file_list = os.listdir(config.new_address_path + f"{prefix}")
#     #     for file in file_list:
#     #         os.remove(config.new_address_path + f"{prefix}/{file}")
#
#     # save new_address with labels to file
#     # new_address_with_labels is a dict
#     df_new_address_label = pd.DataFrame(list(new_address_with_labels.items()), columns=['address', 'label'])
#     # save only address to file for zmap
#     df_new_address_label["address"].to_csv(config.new_address_path + f"gen_{prefix}.txt",
#                                            sep=',', header=False, index=False)
#     # save address and label to address bank for removing duplicate
#     # if there is no "bank_prefix.txt", create one
#     # if there is "bank_prefix.txt", append new address to the end of the file
#     if not os.path.exists(config.address_bank_path + f"bank_{prefix}.txt"):
#         df_new_address_label.to_csv(config.address_bank_path + f"bank_{prefix}.txt",
#                                     sep=',', header=False, index=False)
#     else:
#         with open(config.address_bank_path + f"bank_{prefix}.txt", 'a') as f:
#             df_new_address_label.to_csv(f, sep=',', header=False, index=False)
#
#     # run zmap for prefix
#     hit_rate = run_zmap(prefix)
#
#     # read zmap result and format from string to standard address
#     zmap_active_address = []
#     with open(config.zmap_result_path + f"zmap_gen_{prefix}.txt", 'r') as f:
#         for line in f:
#             zmap_active_address.append(tran_ipv6(line.strip()))
#     df_active_address = pd.DataFrame(zmap_active_address, columns=['address'])
#
#     # remove duplicate from used generated active address (in the form of standard address)
#     df_active_address_label = pd.merge(df_active_address, df_new_address_label, on=['address'])
#     num_active_address = df_active_address_label.shape[0]
#
#     list_alias_prefix = []
#     if df_active_address_label.shape[0] > 1:
#         list_alias_prefix = list_alias_prefix + alias_detection(df_active_address_label, prefix)
#         print('alias prefix :', list_alias_prefix)
#         if list_alias_prefix:
#             # 删除别名地址影响
#             for alias_prefix in list_alias_prefix:
#                 df_active_address_label = df_active_address_label.loc[
#                     df_active_address_label['address'].apply(lambda s: re.search(alias_prefix, s) is None)].reset_index(
#                     drop=True)
#             print('active address after del alias:', df_active_address_label.shape[0])
#
#     num_no_alias_active_address = df_active_address_label.shape[0]
#     num_alias_active_address = num_active_address - num_no_alias_active_address
#     num_new_address = df_new_address_label.shape[0]
#     hit_rate_no_alias = round(num_no_alias_active_address / (num_new_address - num_alias_active_address) * 100, 2)
#     print('hit_rate_no_alias: ', hit_rate_no_alias)
#
#     # delete the intermediate file in new_address_path gen_prefix.txt
#     os.remove(config.new_address_path + f"gen_{prefix}.txt")
#     df_active_address_label.to_csv(config.new_address_path + f"no_alias_{prefix}.txt",
#                                    sep=',', header=False, index=False)
#
#     # count the number of active address for each label in a list
#     counter_label = [0] * cluster_num
#     for i in range(cluster_num):
#         counter_label[i] = df_active_address_label.loc[df_active_address_label['label'] == i].shape[0]
#     print(f"Active address number for each label: {counter_label}")
#     # update the cluster info
#     update_cluster_info(prefix, counter_label)
#
#     return hit_rate, hit_rate_no_alias, list_alias_prefix, num_new_address


# def run_zmap(prefix, local_ipv6="2402:f000:6:1401:46a8:42ff:fe43:6d00"):
#     # if not os.path.exists(config.zmap_result_path + f"{prefix}"):
#     #     os.mkdir(config.zmap_result_path + f"{prefix}")
#     generated_address_path = config.generated_address_path + '{prefix}'.format(prefix=prefix)
#     zmap_result_path = config.zmap_result_path + '{prefix}'.format(prefix=prefix)
#     print(f"Running zmap for prefix {prefix}...")
#     cmd = (f"sudo zmap --ipv6-source-ip={local_ipv6} "
#            f"--ipv6-target-file={generated_address_path}.txt "
#            f"-o {zmap_result_path}.txt -M icmp6_echoscan -B 10M --verbosity=0")
#
#     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     hit_rate = re.findall(r"\d+\.?\d*", p.communicate()[1][-10:].decode('utf-8'))
#     print(f"Hit rate: {hit_rate}%")
#
#     return hit_rate
