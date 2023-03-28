import os
import numpy as np
import torch
from utils import load_test_data, get_cluster_num, format_ipv6, check_duplicate_from_train
from cvae import CVAE
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def test(model, test_loader):
    new_address = []
    model.eval()
    with torch.no_grad():
        for batch, c in enumerate(test_loader):
            c = c.float().to(device)
            # generate
            pred_x = model.generate(c).squeeze(1)
            # round pred_x to int and limit to [0, 15]
            pred_x = pred_x.round().int()
            pred_x[pred_x > 15] = 15
            pred_x[pred_x < 0] = 0
            pred_x = pred_x.cpu().numpy()
            # save to new_address
            for i in range(pred_x.shape[0]):
                temp_address = format_ipv6(pred_x[i].tolist())
                new_address.append(temp_address)
    return new_address


def test_specific_model(prefix):
    print(f"Testing model for prefix {prefix}...")
    test_loader = load_test_data(prefix, 5000)
    cluster_num = get_cluster_num(prefix)
    model = CVAE(cluster_num).to(device)
    # load corresponding model
    model.load_state_dict(torch.load(config.model_path + f"{prefix}.pth"))
    new_address = test(model, test_loader)
    # remove duplicate
    new_address = list(set(new_address))
    print(f"New address number before removing duplicate from train data: {len(new_address)}")
    # remove duplicate from train data
    new_address = check_duplicate_from_train(prefix, new_address)
    print(f"New address number: {len(new_address)}")
    # save new_address to file
    # with open(config.new_address_path + f"{prefix}.txt", "w") as f:
    #     for address in new_address:
    #         f.write(f"{address}\n")


def test_multiple_model(prefix_list):
    for prefix in prefix_list:
        test_specific_model(prefix)


def test_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    for prefix in range(prefix_num):
        test_specific_model(prefix)
