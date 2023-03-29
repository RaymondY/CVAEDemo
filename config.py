import os
import torch
# from datetime import datetime

# cluster_num = -1


class DefaultConfig(object):
    # general
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    # dir path
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data/')
    model_path = os.path.join(current_path, 'model/')
    new_address_path = os.path.join(current_path, 'new_address/')
    # cluster_num_path = os.path.join(current_path, 'cluster_num.txt')
    cluster_info_path = os.path.join(current_path, 'cluster_info.txt')

    input_size = 32
    latent_size = 8

    eps_threshold = 8
    eps_ratio = 0.02
    eps_min_sample = 200

    batch_size = 64
    test_batch_size = 100

    beta = 1
    block_num = 8
    feature_num = 128
    intermediate_size = 128

    epoch_num = 40
    lr = 1e-3

    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = 'logs/gradient_tape/' + current_time
