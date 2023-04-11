import os
from utils import *
from train import *
from evaluate import *
from config import DefaultConfig

config = DefaultConfig()


def init_model_bank():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    # init init_cluster_info_path
    with open(config.init_cluster_info_path, 'w') as f:
        for i in range(prefix_num):
            f.write('-1\n')
    for prefix in range(0, prefix_num):
        init_train_specific_prefix(prefix)


def main():
    pass


if __name__ == '__main__':
    init_model_bank()
    # main()
