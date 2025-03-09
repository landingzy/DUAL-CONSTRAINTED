

import argparse
import numpy as np
import pandas as pd
import os

from ctr import config


def cal_CDF():
    train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.bid.lin.csv'))

    header = ['clk', 'pctr', 'market_price', 'day']

    if config['time_fraction'] == 96:
        header.append('96_time_fraction')
    elif config['time_fraction'] == 48:
        header.append('48_time_fraction')
    elif config['time_fraction'] == 24:
        header.append('24_time_fraction')

    train_data = train_data[header]
    # 0:clk, 1:pctr, 2:market_price, 3:day, 4:time_fraction
    train_data.columns = [0, 1, 2, 3, 4]

    # 转换数据格式
    train_data.iloc[:, [0, 2, 3, 4]] = train_data.iloc[:, [0, 2, 3, 4]].astype(int)
    train_data.iloc[:, [1]] = train_data.iloc[:, [1]].astype(float)

    scaled_ctr=[]
    original_ctr=[]
    normalized_ctr=[]
    # 假设原CTR值存储在train_data[1]中
    original_ctr = train_data.iloc[:, 1].values

    # 计算归一化CTR
    min_original_ctr = np.min(original_ctr)
    max_original_ctr = np.max(original_ctr)
    normalized_ctr = (original_ctr - min_original_ctr) / (max_original_ctr - min_original_ctr)

    # 定义所需的估值范围
    min_desired_value = 0  # 所需范围的最小值
    max_desired_value = 300 # 所需范围的最大值

    # 进行尺度调整
    scaled_ctr = normalized_ctr * (max_desired_value - min_desired_value) + min_desired_value

    columns = ['value']

    d_path = os.path.join(config['data_path'], config['campaign_id'])
    # scaled_ctr = pd.DataFrame(data=scaled_ctr )
    # if not os.path.exists(d_path):
    #     os.makedirs(d_path)

    scaled_ctr = pd.DataFrame(data=scaled_ctr , columns=columns)
    scaled_ctr.to_csv(d_path + '/Fineall_value' + '.csv')
    # scaled_ctr.to_csv(os.path.join(data_path, 'train.bid.lin.csv'), index=None)
    # scaled_ctr 现在包含了调整后的CTR值
