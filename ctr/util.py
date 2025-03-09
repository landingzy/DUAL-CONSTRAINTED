import matplotlib.pyplot as plt

from scipy import stats
import argparse
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KernelDensity


# 求解每个pctr对应的value值存入train.csv文件中
def cal_value():
    train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.bid.lin.csv'))
    print(config)
    # train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'test.bid.lin.csv'))

    train_data['pctr'] = train_data['pctr'].astype(float)
    total_cost=sum(train_data['market_price'])
    total_clk=sum(train_data['clk'])
    clk_value = total_cost / total_clk #cpc
    print(clk_value)

    # 自定义映射函数
    # def custom_mapping_function(click_through_rate, min_value=0, max_value=300):
    #     log_transformed_value = np.log1p(10000 * click_through_rate)
    #     mapped_value = log_transformed_value * (max_value - min_value) / 10
    #     return mapped_value
    def custom_mapping_function(click_through_rate, min_value=0, max_value=300):
        scaled_value = click_through_rate * clk_value
        scaled_value = np.ceil(scaled_value).astype(int)
        mapped_value = min(max(scaled_value, min_value), max_value)
        return mapped_value

    # 将映射后的值存入 'value' 列
    train_data['value'] = train_data['pctr'].apply(custom_mapping_function)


    # 3. 将 d 列插入到 "pctr" 列的后面
    train_data = train_data[
        ['clk', 'pctr', 'value', 'market_price', 'minutes', '24_time_fraction', '48_time_fraction', '96_time_fraction','1440_time_fraction',
         'day']]

    # 4. 另存为 train.csv 文件
    train_data.to_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.csv'), index=False)

    # scaled_ctr 现在包含了调整后的CTR值


def cal_value1():
    train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'test.bid.lin.csv'))

    train_data['pctr'] = train_data['pctr'].astype(float)
    total_cost = sum(train_data['market_price'])
    total_clk = sum(train_data['clk'])
    clk_value = total_cost / total_clk  # cpc
    print(clk_value)
    # 自定义映射函数
    # def custom_mapping_function(click_through_rate, min_value=0, max_value=300):
    #     log_transformed_value = np.log1p(10000 * click_through_rate)
    #     mapped_value = log_transformed_value * (max_value - min_value) / 10
    #     return mapped_value
    def custom_mapping_function(click_through_rate, min_value=0, max_value=300):
        scaled_value = click_through_rate * clk_value
        scaled_value = np.ceil(scaled_value).astype(int)
        mapped_value = min(max(scaled_value, min_value), max_value)
        return mapped_value

    # 将映射后的值存入 'value' 列
    train_data['value'] = train_data['pctr'].apply(custom_mapping_function)


    # 3. 将 d 列插入到 "pctr" 列的后面
    train_data = train_data[
        ['clk', 'pctr', 'value', 'market_price', 'minutes', '24_time_fraction', '48_time_fraction', '96_time_fraction','1440_time_fraction',
         'day']]

    # 4. 另存为 train.csv 文件
    train_data.to_csv(os.path.join(config['data_path'], config['campaign_id'], 'test.csv'), index=False)


# 将每个pctr对应的value存入 rlb中
# def rlb():
#     origin_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'test.csv'))
#     rlb_data = origin_data[['clk', 'pctr', 'value', 'market_price']]
#     data_path = os.path.join(config['data_path'], config['campaign_id'])
#     fout = open(os.path.join(data_path, 'value_train.bid.rlb.txt'), 'w')
#     for index, row in rlb_data.iterrows():
#         fout.write(str(int(row['clk'])) + " " + str(int(row['market_price'])) + " " + str(row['pctr']) + " " + str(
#             row['value']) + '\n')
#     fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ipinyou')
    parser.add_argument('--campaign_id', type=str, default='1458/',
                        help='1458, 2259, 2261, 2821, 2997, 3358, 3427, 3476')

    args = parser.parse_args()
    config = vars(args)

    cal_value()
    cal_value1()
    #rlb()
