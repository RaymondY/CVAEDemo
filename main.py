import os
# from utils import init_cluster_num
from train import train_specific_model, train_multiple_model, train_all_model
from evaluate import test_specific_model, test_multiple_model, test_all_model
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def main():
    # detect if cluster_num.txt exists
    # if not os.path.exists(config.cluster_num_path):
    #     init_cluster_num()
    train_specific_model(400)
    test_specific_model(400)


if __name__ == '__main__':
    main()

    # import numpy as np
    # from utils import init_cluster_info, update_cluster_info
    # init_cluster_info()
    # label_info = [2, 1, 3]
    # update_cluster_info(1, label_info)

    # from utils import load_test_data
    # test_loader = load_test_data(1)
    # print(test_loader.dataset.get_cluster_labels())

    # import numpy as np
    # from utils import format_ipv6
    # path = config.data_path + '{prefix_index}.txt'.format(prefix_index=1)
    # data = np.loadtxt(path, delimiter=',').astype(np.int32)
    # train_ipv6_list = set()
    # for i in range(data.shape[0]):
    #     # format the ipv6 address
    #     temp_ipv6 = format_ipv6(data[i].tolist())
    #     train_ipv6_list.add(temp_ipv6)
    # # save to file
    # with open(config.new_address_path + f"data1.txt", "w") as f:
    #     for address in train_ipv6_list:
    #         f.write(f"{address}\n")
