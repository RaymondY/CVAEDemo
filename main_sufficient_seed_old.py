# su seeds  直接在本身有种子前缀下(模型下)继续生成
import os
import pandas as pd
from utils import init_cluster_info, copy_model
from evaluate import test_specific_model
from train import train_specific_model, fine_tuning_specific_model
from config import DefaultConfig

config = DefaultConfig()


def init_probe_specific_model(prefix, budget):
    hit_rate, hit_rate_no_alias, alias_prefix_list, num_init_probe_address = test_specific_model(prefix, sample_num=budget)
    return hit_rate, hit_rate_no_alias, alias_prefix_list


def init_probe_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    prefix_hit_alias_list = []
    for prefix in range(0, 354):
    # for prefix in range(2):
        print(f"**************** init probe prefix {prefix} ...")
        hit_rate, hit_rate_no_alias, alias_prefix_list, num_init_probe_address = test_specific_model(prefix, config.prefix_budget * 0.1)
        prefix_hit_alias_list.append([prefix, hit_rate, hit_rate_no_alias, alias_prefix_list, num_init_probe_address])
    df_prefix_hit_alias = pd.DataFrame(data=prefix_hit_alias_list,
                                       columns=['prefix', 'hit_rate', 'hit_rate_no_alias', 'alias_prefix', 'num_init_probe_address'])
    path = config.zmap_result_path + 'prefix_hit_alias_info.txt'
    df_prefix_hit_alias.to_csv(path, index=False)


def iter_probe_specific_model(prefix, budget):

    path = config.zmap_result_path + 'prefix_hit_alias_info.txt'
    df_prefix_hit_alias = pd.read_csv(path)

    # copy model to fining
    copy_model(prefix)
    fine_lower_limit = 50
    df_all_active_address = pd.DataFrame(data = None)  # 记录所有活跃地址
    df_prefix_init_info = df_prefix_hit_alias[df_prefix_hit_alias['prefix'] == prefix]

    iter_info_record = [df_prefix_hit_alias[df_prefix_hit_alias['prefix'] == prefix].values.tolist()]

    path = config.new_address_path + 'no_alias_{prefix}.txt'.format(prefix=prefix)
    hit_rate_no_alias = 1
    while hit_rate_no_alias > 0.0:
        i = 0
        df_active_address_label = pd.read_csv(path, sep=',')
        num_active = df_active_address_label.shape[0]
        df_all_active_address.append(df_active_address_label)

        if df_active_address_label.shape[0] > fine_lower_limit:
            # Fine tuning
            print(f"Fine tuning model for {i + 1} time...")
            # 读取带标签数据 并 微调模型 读取模型就从model_fine
            fine_tuning_specific_model(prefix)

        # 查看总预算是否够
        temp_buget = num_active
        if temp_buget > budget:
            growth_factor = budget / temp_buget
        elif temp_buget * config.growth_factor > budget:
            growth_factor = budget / temp_buget
        else:
            growth_factor = config.growth_factor

        hit_rate, hit_rate_no_alias, alias_prefix_list, num_probe_address = test_specific_model(prefix, num_active * growth_factor)
        iter_info_record.append([prefix, hit_rate, hit_rate_no_alias, alias_prefix_list, num_init_probe_address])
        budget = budget - num_probe_address

    df_all_active_address.to_csv(path)
    



def iter_probe_all_model(budget):
    path = config.zmap_result_path + 'prefix_hit_alias_info.txt'
    df_prefix_hit_alias = pd.read_csv(path)
    df_prefix_hit_alias = \
        df_prefix_hit_alias.sort_values(by=['hit_rate_no_alias'], ascending=False).reset_index(drop=True)
    for i in range(df_prefix_hit_alias.shape[0]):
        prefix = df_prefix_hit_alias.loc[i, 'prefix']
        alias_prefix = df_prefix_hit_alias.loc[i, 'alias_prefix']
        iter_probe_specific_model(prefix, budget, alias_prefix)


def main():# detect if cluster_info.txt exists
    if not os.path.exists(config.cluster_info_path):
        init_cluster_info()


if __name__ == '__main__':
    test_prefix = 1
    # budget = config.prefix_budget * 0.1
    # hitrate, no_alias_hitrate, list_alias_prefix = init_probe_specific_model(test_prefix, budget)
    budget = config.prefix_budget * 0.9
    iter_probe_specific_model(test_prefix, budget)
    # init_probe_all_model()
