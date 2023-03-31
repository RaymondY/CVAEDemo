# su seeds  直接在本身有种子前缀下(模型下)继续生成
import os
import pandas as pd
from utils import init_cluster_info, copy_model
from train import train_specific_model, train_multiple_model, train_all_model, \
    fine_tuning_specific_model, fine_tuning_multiple_model, fine_tuning_all_model
from evaluate import test_specific_model, test_multiple_model, test_all_model
from config import DefaultConfig

config = DefaultConfig()


def init_su_seed_specific_model(prefix, budget):
    hit_rate, hit_rate_no_alias, alias_prefix_list = test_specific_model(prefix, sample_num=budget)
    return hit_rate, hit_rate_no_alias, alias_prefix_list


def init_su_seed_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    prefix_hit_alias_list = []
    for prefix in range(prefix_num):
        hit_rate, hit_rate_no_alias, alias_prefix_list = test_specific_model(prefix, config.prefix_budget * 0.1)
        prefix_hit_alias_list = prefix_hit_alias_list + [prefix, hit_rate, hit_rate_no_alias, alias_prefix_list]
    df_prefix_hit_alias = pd.DataFrame(data=prefix_hit_alias_list,
                                       columns=['prefix', 'hit_rate', 'hit_rate_no_alias', 'alias_prefix'])
    path = config.zmap_result_path + 'prefix_hit_alias_info.txt'
    df_prefix_hit_alias.to_csv(path, index=False)


def iter_su_seed_specific_model(prefix, budget, alias_prefix):
    path = config.zmap_result_path + 'no_alias_{prefix}.txt'.format(prefix=prefix)
    df_active_address_label = pd.read_csv(path, sep=',')

    # copy model to fining
    copy_model(prefix)
    fine_lower_limit = 50
    df_all_active_address = df_active_address_label["address"]  # 记录所有活跃地址
    while float(hit_rate_prob) > 0.0:
        i = 0
        if df_active_address_label.shape[0] > fine_lower_limit:
            # Fine tuning
            print(f"Fine tuning model for {i + 1} time...")
            # 读取带标签数据 并 微调模型 读取模型就从model_fine
            fine_tuning_specific_model(prefix, load_model_path=config.model_fined_path)

        # budget allocation
        counter_label = df_active_address_label['label'].value_counts().to_dict()

        temp_buget = sum(counter_label.values())
        if temp_buget > budget:
            growth_factor = budget / temp_buget
        elif temp_buget * config.growth_factor > budget:
            growth_factor = budget / temp_buget
        else:
            growth_factor = config.growth_factor

        # 每个label 预算 不再是依据聚类分配，而是依据counter_label,  此轮总预算为 上次活跃地址数 * growth_factor  
        # 去重不仅原始数据还有df_all_active_address
        test_iter_specific_model(prefix, counter_label,
                                 df_all_active_address)  # test_specific_model(prefix, sample_num)


def iter_su_seed_all_model(prefix, budget):
    path = config.zmap_result_path + 'prefix_hit_alias_info.txt'
    df_prefix_hit_alias = pd.read_csv(path)
    df_prefix_hit_alias = \
        df_prefix_hit_alias.sort_values(by=['hit_rate_no_alias'], ascending=False).reset_index(drop=True)
    for i in range(df_prefix_hit_alias.shape[0]):
        prefix = df_prefix_hit_alias.loc[i, 'prefix']
        alias_prefix = df_prefix_hit_alias.loc[i, 'alias_prefix']
        iter_su_seed_specific_model(prefix, budget, alias_prefix)


def main():# detect if cluster_info.txt exists
    if not os.path.exists(config.cluster_info_path):
        init_cluster_info()


if __name__ == '__main__':
    test_prefix = 1
    budget = config.prefix_budget * 0.1
    hitrate, no_alias_hitrate, list_alias_prefix = init_su_seed_specific_model(test_prefix, budget)
    # budget = config.prefix_budget * 0.9
    # iter_su_seed_specific_model(prefix, budget, list_alias_prefix)
