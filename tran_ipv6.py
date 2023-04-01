import os
import csv
# from xml.dom import INVALID_MODIFICATION_ERR
import pandas as pd
from collections import Counter
import pyasn
# from math import log, e
import numpy as np


def read_file(folder, file):
    ip_list = []
    filename = folder + file
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ip = row[0]
            ip_list.append(ip)
    return ip_list


def tran_ipv6(sim_ip):
    if sim_ip == "::":
        return "0000:0000:0000:0000:0000:0000:0000:0000"
    ip_list = ["0000", "0000", "0000", "0000", "0000", "0000", "0000", "0000"]
    if sim_ip.startswith("::"):
        temp_list = sim_ip.split(":")
        for i in range(0, len(temp_list)):
            ip_list[i + 8 - len(temp_list)] = ("0000" + temp_list[i])[-4:]
    elif sim_ip.endswith("::"):
        temp_list = sim_ip.split(":")
        for i in range(0, len(temp_list)):
            ip_list[i] = ("0000" + temp_list[i])[-4:]
    elif "::" not in sim_ip:
        temp_list = sim_ip.split(":")
        for i in range(0, len(temp_list)):
            ip_list[i] = ("0000" + temp_list[i])[-4:]
    # elif sim_ip.index("::") > 0:
    else:
        temp_list = sim_ip.split("::")
        temp_list0 = temp_list[0].split(":")
        # print(temp_list0)
        for i in range(0, len(temp_list0)):
            ip_list[i] = ("0000" + temp_list0[i])[-4:]
        temp_list1 = temp_list[1].split(":")
        # print(temp_list1)
        for i in range(0, len(temp_list1)):
            ip_list[i + 8 - len(temp_list1)] = ("0000" + temp_list1[i])[-4:]
    # else:
    #     temp_list = sim_ip.split(":")
    #     for i in range(0, temp_list):
    #         ip_list[i] = ("0000" + temp_list[i])[-4:]
    # print(ip_list)
    return ":".join(ip_list)


# def pandas_entropy(column, base=None):
def pandas_entropy(column):
    # vc = pd.Series(column).value_counts(normalize=True, sort=False)
    # base = e if base is None else base
    # return -(vc * np.log(vc)/np.log(base)).sum()
    a = pd.value_counts(column) / len(column)
    return sum(np.log2(a) * a * (-1))


def data_pre(folder):
    filelist = os.listdir(folder)
    # print(filelist)
    filelist.sort()
    # number = len(filelist)
    # print(number,'/1280',' ',number/1280,'%')
    ip_list = []
    for file in filelist:
        # file = "test2.csv"
        list_a = read_file(folder, file)
        # print(list)
        for i, val in enumerate(list_a):
            ip_list.append(tran_ipv6(val))

    ip_df = pd.DataFrame(ip_list, columns=['address'])

    # remove_duplicate_addresses
    ip_df.drop_duplicates(subset=ip_df.columns, keep='last', inplace=True)
    print("*" * 20 + "number of seed addresses: ", ip_df.shape)
    ip_df["prefix_pyasn"] = None

    asndb = pyasn.pyasn('ipasn.20220916.dat')

    for index, row in ip_df.iterrows():
        prefix_pyasn = asndb.lookup(row['address'])[1]
        row["prefix_pyasn"] = prefix_pyasn

    ip_df_group = ip_df.groupby('prefix_pyasn')
    # print(ip_df_group)

    # suitable prefix
    ip_prefix_list = ip_df["prefix_pyasn"].to_list()
    result = Counter(ip_prefix_list)
    result_list = []
    seed_num = []
    print()
    for k, v in result.items():
        seed_num.append([k, v])
        if v >= 1000:
            result_list.append(k)

    list_id_as = []
    for prefix_id in range(len(result_list)):
        print(f"prefix_id: {prefix_id}")
        prefix_group = ip_df_group.get_group(result_list[prefix_id])
        print(prefix_group.shape[0])

        # 获取AS号
        prefix_as = asndb.lookup(result_list[prefix_id][0:-3])[0]
        list_id_as.append([prefix_id, prefix_as])

        # # 采样前每个前缀原始数据保存
        # path = config.data_path + 'ipv6___' + '{prefix_id}.txt'.format(prefix_id=prefix_id)
        # prefix_group.to_csv(path, header=False, index=False)

        # *****************random sampling 1000
        # prefix_group = prefix_group.sample(1000, random_state=88)
        # print(prefix_group.head())

        # # 采样后每个前缀原始数据保存
        # path = config.data_path + 'ipv6_' + '{prefix_id}.txt'.format(prefix_id=prefix_id)
        # prefix_group["address"].to_csv(path, header=False, index=False)

        prefix_group = prefix_group["address"].str.replace(':', '').astype(str).to_frame()

        list_all = []
        for index, row in prefix_group.iterrows():
            list_temp = list(row["address"])
            list_all.append(list_temp)
        df_entropy = pd.DataFrame(data=list_all)

        # for i in range(32):
        #     per_entro = pandas_entropy(df_entropy.iloc[:, i], base=None)
        #     # print(per_entro)
        #     if per_entro > 3.5:
        #         df_entropy.iloc[:, i] = '*'

        list_all = []
        for index, row in df_entropy.iterrows():
            list_temp = row.to_list()
            for j in range(32):
                if list_temp[j] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    continue
                elif list_temp[j] == '*':
                    list_temp[j] = '16'
                elif list_temp[j] == 'a':
                    list_temp[j] = '10'
                elif list_temp[j] == 'b':
                    list_temp[j] = '11'
                elif list_temp[j] == 'c':
                    list_temp[j] = '12'
                elif list_temp[j] == 'd':
                    list_temp[j] = '13'
                elif list_temp[j] == 'e':
                    list_temp[j] = '14'
                elif list_temp[j] == 'f':
                    list_temp[j] = '15'
            list_all.append(list_temp)

        prefix_group = pd.DataFrame(data=list_all)
        # path = './su_' + '{prefix_id}.txt'.format(prefix_id=prefix_id)
        path = 'data/' + '{prefix_id}.txt'.format(prefix_id=prefix_id)
        prefix_group.to_csv(path, sep=',', header=False, index=False)


'''
读取 600多万条数据
完成地址的格式转换(tran_ipv6)
然后按照阈值(104行)大小(即只有一个前缀下地址数量大于阈值才会输出)
分前缀分别输出到su_**.txt
'''
if __name__ == "__main__":
    # 把scan_output_responsive.txt 放到这个路径下即可
    # folder = '/home/chengdaguo/IPv6_scan/scanipv6/data/test_github_1/data_activate/'  #600多万数据路径
    folder = "./raw_data/"
    data_pre(folder)

# 使用zmap工具在线扫描ipv6地址是否活跃 1. 安装zmap工具 2. 命令行执行(把ipv6源地址修改为自己的)： XX都可以是.txt文件 sudo zmap
# --ipv6-source-ip=2402:f000:6:1401::226 --ipv6-target-file=XX  -o result_file(XX)  -M icmp6_echoscan -B 10M
# --interface=eno4 --verbosity=0


# 熵处理版本，如果一位熵值大于3.5 则置为 *

# def data_pre(folder):
#     filelist = os.listdir(folder)
#     # print(filelist)
#     filelist.sort()
#     number = len(filelist)
#     # print(number,'/1280',' ',number/1280,'%')
#     ip_list = []
#     for file in filelist:
#         # file = "test2.csv"
#         list_a = read_file(folder, file)                                                                                                                                                                                                                  
#         # print(list)
#         for i, val in enumerate(list_a):
#                 ip_list.append(tran_ipv6(val))

#     ip_df = pd.DataFrame(ip_list, columns=['address'])

#     #remove_duplicate_addresses
#     ip_df.drop_duplicates(subset = ip_df.columns,keep='last',inplace=True)
#     print("*"*20 + "number of seed addresses: ", ip_df.shape)
#     ip_df["prefix_pyasn"] = None

#     asndb = pyasn.pyasn('ipasn.20220916.dat')

#     for index, row in ip_df.iterrows():
#         prefix_pyasn = asndb.lookup(row['address'])[1]
#         row["prefix_pyasn"] = prefix_pyasn

#     ip_df_group = ip_df.groupby('prefix_pyasn')
#     # print(ip_df_group)

#     # suitable prefix
#     ip_prefix_list = ip_df["prefix_pyasn"].to_list()
#     result = Counter(ip_prefix_list)
#     result_list = []
#     seed_num = []
#     for k, v in result.items():
#         seed_num.append([k, v])
#         if v>=1000:
#             result_list.append(k)  

#     list_id_as = []
#     for prefix_id in range(len(result_list)):
#         prefix_group = ip_df_group.get_group(result_list[prefix_id])
#         print(prefix_group.shape[0])

#         #获取AS号
#         prefix_as = asndb.lookup(result_list[prefix_id][0:-3])[0]
#         list_id_as.append([prefix_id, prefix_as])

#         # # 采样前每个前缀原始数据保存
#         # path = config.data_path + 'ipv6___' + '{prefix_id}.txt'.format(prefix_id=prefix_id)
#         # prefix_group.to_csv(path, header=False, index=False)

#         prefix_group = prefix_group.sample(1000, random_state=88)
#         print(prefix_group.head())

#         # # 采样后每个前缀原始数据保存
#         # path = config.data_path + 'ipv6_' + '{prefix_id}.txt'.format(prefix_id=prefix_id)
#         # prefix_group["address"].to_csv(path, header=False, index=False)

#         prefix_group = prefix_group["address"].str.replace(':','').astype(str).to_frame()


#         list_all = []
#         for index, row in prefix_group.iterrows():
#             list_temp = list(row["address"])
#             list_all.append(list_temp)
#         df_entropy = pd.DataFrame(data=list_all)

#         for i in range(32):
#             per_entro = pandas_entropy(df_entropy.iloc[:,i], base=None)
#             # print(per_entro)
#             if per_entro > 3.5:
#                 df_entropy.iloc[:,i] = '*'

#         list_all = []
#         for index, row in df_entropy.iterrows():
#             list_temp = row.to_list()
#             for j in range(32):
#                 if list_temp[j] in ['0','1','2','3','4','5','6','7','8', '9']:
#                     continue
#                 elif list_temp[j] == '*':
#                     list_temp[j] = '16'
#                 elif list_temp[j] == 'a':
#                     list_temp[j] = '10'
#                 elif list_temp[j] == 'b':
#                     list_temp[j] = '11'
#                 elif list_temp[j] == 'c':
#                     list_temp[j] = '12'
#                 elif list_temp[j] == 'd':
#                     list_temp[j] = '13'
#                 elif list_temp[j] == 'e':
#                     list_temp[j] = '14'
#                 elif list_temp[j] == 'f':
#                     list_temp[j] = '15'
#             list_all.append(list_temp)


#         prefix_group = pd.DataFrame(data=list_all)
#         # # # prefix_group.drop_duplicates(subset = prefix_group.columns.values,keep='last',inplace=True)
#         path = './su_' + '{prefix_id}.txt'.format(prefix_id=prefix_id)
#         prefix_group.to_csv(path, sep=',', header=False, index=False)
