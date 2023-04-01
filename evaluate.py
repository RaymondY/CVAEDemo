import os
import re
import subprocess
import pandas as pd
import torch
from config import DefaultConfig
from cvae import CVAE
from utils import load_test_data, get_cluster_num, check_duplicate_from_train, check_duplicate_from_bank, \
    format_vector_to_standard, format_str_to_vector, format_str_to_standard, alias_detection, update_cluster_info, \
    tran_ipv6

config = DefaultConfig()
device = config.device


def test(model, test_loader):
    # return new address with cluster label together
    new_address_with_label = dict()
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
                if temp_address not in new_address_with_label:
                    new_address_with_label[temp_address] = temp_label
                elif new_address_with_label[temp_address] != temp_label:
                    print(f"Error: {temp_address} has different cluster label.")
                    print(f"Old label: {new_address_with_label[temp_address]}, new label: {temp_label}")
                    new_address_with_label[temp_address] = temp_label
    return new_address_with_label


def test_specific_model(prefix, sample_num=5000):
    print(f"Testing model for prefix {prefix}...")
    test_loader = load_test_data(prefix, sample_num)
    cluster_num = get_cluster_num(prefix)
    model = CVAE(cluster_num).to(device)
    # load corresponding model
    model.load_state_dict(torch.load(config.model_path + f"{prefix}.pth"))
    # the new address saved in a dict
    new_address_with_labels = test(model, test_loader)
    print(f"New address number before removing duplicate from train data: {len(new_address_with_labels)}")
    # remove duplicate from train data
    check_duplicate_from_train(prefix, new_address_with_labels)
    print(f"New address number before removing duplicate from address bank: {len(new_address_with_labels)}")
    # remove duplicate from address bank
    check_duplicate_from_bank(prefix, new_address_with_labels)
    print(f"New address number: {len(new_address_with_labels)}")

    # mkdir "prefix" if not exists
    # if not os.path.exists(config.new_address_path + f"{prefix}"):    os.path.exists(config.zmap_result_path
    #     os.mkdir(config.new_address_path + f"{prefix}")
    # else:
    #     # remove all files in the folder, in case of mixing up old and new files
    #     file_list = os.listdir(config.new_address_path + f"{prefix}")
    #     for file in file_list:
    #         os.remove(config.new_address_path + f"{prefix}/{file}")

    # save new_address with labels to file
    # new_address_with_labels is a dict
    df_new_address_label = pd.DataFrame(list(new_address_with_labels.items()), columns=['address', 'label'])
    # save only address to file for zmap
    df_new_address_label["address"].to_csv(config.new_address_path + f"gen_{prefix}.txt",
                                           sep=',', header=False, index=False)
    # save address and label to address bank for removing duplicate
    # if there is no "bank_prefix.txt", create one
    # if there is "bank_prefix.txt", append new address to the end of the file
    if not os.path.exists(config.address_bank_path + f"bank_{prefix}.txt"):
        df_new_address_label.to_csv(config.address_bank_path + f"bank_{prefix}.txt",
                                    sep=',', header=False, index=False)
    else:
        with open(config.address_bank_path + f"bank_{prefix}.txt", 'a') as f:
            df_new_address_label.to_csv(f, sep=',', header=False, index=False)

    # run zmap for prefix
    hit_rate = run_zmap(prefix)

    # read zmap result and format from string to standard address
    zmap_active_address = []
    with open(config.zmap_result_path + f"zmap_gen_{prefix}.txt", 'r') as f:
        for line in f:
            zmap_active_address.append(tran_ipv6(line.strip()))
    df_active_address = pd.DataFrame(zmap_active_address, columns=['address'])

    # remove duplicate from used generated active address (in the form of standard address)
    df_active_address_label = pd.merge(df_active_address, df_new_address_label, on=['address'])
    num_active_address = df_active_address_label.shape[0]

    list_alias_prefix = []
    if df_active_address_label.shape[0] > 1:
        list_alias_prefix = list_alias_prefix + alias_detection(df_active_address_label, prefix)
        print('alias prefix :', list_alias_prefix)
        if list_alias_prefix:
            # 删除别名地址影响
            for alias_prefix in list_alias_prefix:
                df_active_address_label = df_active_address_label.loc[
                    df_active_address_label['address'].apply(lambda s: re.search(alias_prefix, s) is None)].reset_index(
                    drop=True)
            print('active address after del alias:', df_active_address_label.shape[0])

    num_no_alias_active_address = df_active_address_label.shape[0]
    num_alias_active_address = num_active_address - num_no_alias_active_address
    num_new_address = df_new_address_label.shape[0]
    hit_rate_no_alias = round(num_no_alias_active_address / (num_new_address - num_alias_active_address) * 100, 2)
    print('hit_rate_no_alias: ', hit_rate_no_alias)

    # delete the intermediate file in new_address_path gen_prefix.txt
    os.remove(config.new_address_path + f"gen_{prefix}.txt")
    df_active_address_label.to_csv(config.new_address_path + f"no_alias_{prefix}.txt",
                                   sep=',', header=False, index=False)

    # count the number of active address for each label in a list
    counter_label = [0] * cluster_num
    for i in range(cluster_num):
        counter_label[i] = df_active_address_label.loc[df_active_address_label['label'] == i].shape[0]
    print(f"Active address number for each label: {counter_label}")
    # update the cluster info
    update_cluster_info(prefix, counter_label)

    return hit_rate, hit_rate_no_alias, list_alias_prefix, num_new_address


def run_zmap(prefix, local_ipv6="2402:f000:6:1401:46a8:42ff:fe43:6d00"):
    # if not os.path.exists(config.zmap_result_path + f"{prefix}"):
    #     os.mkdir(config.zmap_result_path + f"{prefix}")
    new_address_dir = config.new_address_path + 'gen_{prefix}'.format(prefix=prefix)
    zmap_result_dir = config.zmap_result_path + 'zmap_gen_{prefix}'.format(prefix=prefix)
    print(f"Running zmap for prefix {prefix}...")
    cmd = (f"sudo zmap --ipv6-source-ip={local_ipv6} "
           f"--ipv6-target-file={new_address_dir}.txt "
           f"-o {zmap_result_dir}.txt -M icmp6_echoscan -B 10M --verbosity=0")

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    hit_rate = re.findall(r"\d+\.?\d*", p.communicate()[1][-10:].decode('utf-8'))
    print(f"Hit rate: {hit_rate}%")

    return hit_rate
