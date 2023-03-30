import os
import numpy as np
import torch
from utils import load_test_data, get_cluster_num, format_ipv6, check_duplicate_from_train
from cvae import CVAE
from config import DefaultConfig

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
                temp_address = format_ipv6(pred_x[i].tolist())
                # turn one-hot to label
                temp_label = c[i].argmax().item()
                if temp_address not in new_address_with_label:
                    new_address_with_label[temp_address] = temp_label
                elif new_address_with_label[temp_address] != temp_label:
                    print(f"Error: {temp_address} has different cluster label.")
                    print(f"Old label: {new_address_with_label[temp_address]}, new label: {temp_label}")
                    new_address_with_label[temp_address] = temp_label
    return new_address_with_label


def test_specific_model(prefix):
    print(f"Testing model for prefix {prefix}...")
    test_loader = load_test_data(prefix)
    cluster_num = get_cluster_num(prefix)
    model = CVAE(cluster_num).to(device)
    # load corresponding model
    model.load_state_dict(torch.load(config.model_path + f"{prefix}.pth"))
    # the new address saved in a dict
    new_address_with_labels = test(model, test_loader)
    print(f"New address number before removing duplicate from train data: {len(new_address_with_labels)}")
    # remove duplicate from train data
    check_duplicate_from_train(prefix, new_address_with_labels)
    print(f"New address number: {len(new_address_with_labels)}")
    # # save new_address to file
    # with open(config.new_address_path + f"{prefix}.txt", "w") as f:
    #     for address in new_address:
    #         f.write(f"{address}\n")

    # mkdir "prefix" if not exists
    if not os.path.exists(config.new_address_path + f"{prefix}"):
        os.mkdir(config.new_address_path + f"{prefix}")
    else:
        # remove all files in the folder, in case of mixing up old and new files
        file_list = os.listdir(config.new_address_path + f"{prefix}")
        for file in file_list:
            os.remove(config.new_address_path + f"{prefix}/{file}")

    # save new_address to different file according to cluster label
    for i in range(cluster_num):
        with open(config.new_address_path + f"{prefix}/{i}.txt", "w") as f:
            for address, label in new_address_with_labels.items():
                if label == i:
                    f.write(f"{address}\n")

    # run zmap for each cluster
    run_zmap(prefix)


def test_multiple_model(prefix_list):
    for prefix in prefix_list:
        test_specific_model(prefix)


def test_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    for prefix in range(prefix_num):
        test_specific_model(prefix)


def run_zmap(prefix, local_ipv6="2402:f000:6:1401:46a8:42ff:fe43:6d00"):
    if not os.path.exists(config.result_path + f"{prefix}"):
        os.mkdir(config.result_path + f"{prefix}")
    result_dir = config.result_path + '{prefix}/'.format(prefix=prefix)
    new_address_dir = config.new_address_path + '{prefix}/'.format(prefix=prefix)
    cluster_num = os.listdir(new_address_dir).__len__()
    for i in range(cluster_num):
        print(f"Running zmap for prefix {prefix}, cluster {i}...")
        # print result_file path
        print(f"Result file path: {result_dir}output_{i}.txt")
        os.system(f"sudo zmap --ipv6-source-ip={local_ipv6} "
                  f"--ipv6-target-file={new_address_dir}{i}.txt "
                  f"-o {result_dir}output_{i}.txt -M icmp6_echoscan -B 10M")
