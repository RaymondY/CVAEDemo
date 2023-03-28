import os
from utils import init_cluster_num
from train import train_specific_model, train_multiple_model, train_all_model
from evaluate import test_specific_model, test_multiple_model, test_all_model
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def main():
    # detect if cluster_num.txt exists
    if not os.path.exists(config.cluster_num_path):
        init_cluster_num()
    # train_specific_model(1)
    test_specific_model(1)


if __name__ == '__main__':
    main()
