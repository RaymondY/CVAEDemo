import os
import torch
from datetime import datetime


class DefaultConfig(object):
    # general
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else device)

    # dir path
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data/')
    model_dir = os.path.join(current_path, 'model')
    model_path = os.path.join(model_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth')
    # sample_num = 1000
    # input_size = 32 * 17
    # input_size = 32
    input_size = 32
    cluster_num = 5
    distance_threshold = 8
    batch_size = 20
    # other_result_path_test = '../result/1109/test1/'
    # entropy_threshold = 3.5

    block_num = 4
    feature_num = 32
    intermediate_size = 16
    latent_size = 5
    # kernel_size_1 = 6  # 12
    # kernel_size_2 = 3  # 6
    # stride_1 = 2  # 4
    # stride_2 = 1  # 3

    epoch_num = 10
    lr = 1e-3
    # batch_size = 18

    # lr_decay = 0.95  # when val_loss increase, lr = lr * lr_decay
    # weight_decay = 1e-10

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/gradient_tape/' + current_time
