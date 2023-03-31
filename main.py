import os
from utils import init_cluster_info
from train import train_specific_model, train_multiple_model, train_all_model, \
    fine_tuning_specific_model, fine_tuning_multiple_model, fine_tuning_all_model
from evaluate import test_specific_model, test_multiple_model, test_all_model
from config import DefaultConfig

config = DefaultConfig()


def main():
    # detect if cluster_info.txt exists
    if not os.path.exists(config.cluster_info_path):
        init_cluster_info()

    # test_prefix = 1
    # iterate_time = 2

    # print("Training model...")
    # train_specific_model(test_prefix)

    # print("Testing model...")
    # test_specific_model(test_prefix)

    # print("Fine tuning model for the first time...")
    # fine_tuning_specific_model(test_prefix)

    # print("Testing fine tuned model...")
    # test_specific_model(test_prefix)

    # # if iterate for more than one time, we need to change the load_model_path
    # for i in range(iterate_time - 1):
    #     print(f"Fine tuning model for {i + 1} time...")
    #     fine_tuning_specific_model(test_prefix, load_model_path=config.model_fined_path)
    #     print("Testing fine tuned model...")
    #     test_specific_model(test_prefix)

    train_all_model()



if __name__ == '__main__':
    main()
