import os
import torch


class DefaultConfig(object):
    # general
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    # dir path
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data/')
    model_path = os.path.join(current_path, 'model/')
    iter_model_path = os.path.join(current_path, 'iter_model/')

    # temp data path
    generated_address_path = os.path.join(current_path, 'generated_address/')
    new_address_path = os.path.join(current_path, 'new_address/')
    address_bank_path = os.path.join(current_path, 'address_bank/')
    zmap_result_path = os.path.join(current_path, 'zmap_result/')
    iter_info_path = os.path.join(current_path, 'iter_info/')

    init_cluster_info_path = os.path.join(current_path, 'init_cluster_info.txt')
    cluster_info_path = os.path.join(current_path, 'cluster_info.txt')

    local_ipv6 = "2402:f000:6:1401:46a8:42ff:fe43:6d00"

    input_size = 32
    latent_size = 8

    eps_threshold = 16
    eps_ratio = 0.01
    eps_min_sample = 100

    batch_size = 64
    test_batch_size = 100

    beta = 1
    block_num = 8
    feature_num = 128
    intermediate_size = 128

    epoch_num = 40
    lr = 1e-3

    warmup_epoch_num = 3
    warmup_lr = 1e-5

    prefix_budget = 5000  
    growth_factor = 10
