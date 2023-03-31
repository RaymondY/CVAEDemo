# su seeds  直接在本身有种子前缀下(模型下)继续生成
import os
import pandas as pd
from utils import init_cluster_info, copy_model
from train import train_specific_model, fine_tuning_specific_model
from evaluate import test_specific_model
from config import DefaultConfig

config = DefaultConfig()


def train_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    for prefix in range(prefix_num):
        train_specific_model(prefix)


def fine_tuning_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    for prefix in range(prefix_num):
        fine_tuning_specific_model(prefix)


def init_probe_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    prefix_hit_alias_list = []
    for prefix in range(prefix_num):
        hit_rate, hit_rate_no_alias, alias_prefix_list = test_specific_model(prefix, config.prefix_budget * 0.1)
        prefix_hit_alias_list.append([prefix, hit_rate, hit_rate_no_alias, alias_prefix_list])
    df_prefix_hit_alias = pd.DataFrame(data=prefix_hit_alias_list,
                                       columns=['prefix', 'hit_rate', 'hit_rate_no_alias', 'alias_prefix'])
    path = config.current_path + 'prefix_hit_alias_info.txt'
    df_prefix_hit_alias.to_csv(path, index=False)


def iter_probe_all_model(overall_budget):
    path = config.current_path + 'prefix_hit_alias_info.txt'
    df_prefix_hit_alias = pd.read_csv(path)
    # init budget per prefix as a list
    budget_per_prefix = [overall_budget / len(df_prefix_hit_alias)]
    # allocate budget according to -------------------
    # NEED TO BE IMPLEMENTED
    # -----------------------------------------------
    for prefix, budget in enumerate(budget_per_prefix):
        hit_rate, hit_rate_no_alias, alias_prefix_list = test_specific_model(prefix, budget)


def main():
    if not os.path.exists(config.cluster_info_path):
        init_cluster_info()


if __name__ == '__main__':
    main()
