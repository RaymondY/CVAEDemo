import os
from utils import init_cluster_info
from train import train_specific_model, train_multiple_model, train_all_model
from evaluate import test_specific_model, test_multiple_model, test_all_model
from config import DefaultConfig

config = DefaultConfig()


def main():
    # detect if cluster_info.txt exists
    if not os.path.exists(config.cluster_info_path):
        init_cluster_info()
    # train_specific_model(400)
    test_specific_model(400)


if __name__ == '__main__':
    main()
