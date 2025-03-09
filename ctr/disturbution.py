import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/ipinyou')
parser.add_argument('--campaign_id', type=str, default='1458', help='1458, 2259, 2261, 2821, 2997, 3358, 3427, 3476')
#
args = parser.parse_args()
config = vars(args)
print(config)
# # 步骤1：加载样本数据
# 假设 'value' 是估值的列名
# df= pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.csv'))
# sample_data = df['pctr'].values

# # 步骤3：估计均值和方差
# estimated_mean = np.mean(sample_data)
# estimated_variance = np.var(sample_data)
#
# # 步骤4：构造正态分布
# normal_distribution = norm(loc=estimated_mean, scale=np.sqrt(estimated_variance))
#
# # 步骤5：计算每个值的 CDF 和 PDF 值
# cdf_values = normal_distribution.cdf(sample_data)
# pdf_values = normal_distribution.pdf(sample_data)
#
# # 步骤6：计算 value - (1 - CDF) / PDF
# adjusted_values = sample_data - (1 - cdf_values) / pdf_values
# # 步骤6：创建 DataFrame
# result_df = pd.DataFrame({'Value': sample_data, 'CDF': cdf_values, 'PDF': pdf_values,'vitrual_value':adjusted_values})
#
#
#
# # 步骤7：保存为 CSV 文件
# result_df.to_csv(os.path.join(config['data_path'], config['campaign_id'], 'vitrual_value.csv'), index=False)
# # 可选：绘制正态分布曲线
# x = np.linspace(min(sample_data), max(sample_data), 100)
# plt.plot(x, normal_distribution.cdf(x), label='Normal Distribution', color='red')
#
# plt.legend()
# plt.show()
# plt.savefig(os.path.join(config['data_path'], config['campaign_id'], '11.png'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 读取数据集
train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.csv'))

# 计算样本均值和样本标准差
mean_value = train_data['value'].mean()
std_dev = train_data['value'].std()

# 构建正态分布
normal_dist = norm(loc=mean_value, scale=std_dev)
#
# # 计算 F(v) 和 f(v)
# train_data['F_v'] = normal_dist.cdf(train_data['value']).round(2)
# train_data['f_v'] = normal_dist.pdf(train_data['value']).round(2)
#
# # 计算虚拟价值函数 Φ(v)
# train_data['Phi_v'] =( train_data['value'] - (1 - train_data['F_v']) / train_data['f_v']).round(2)
#
# # 将结果存储到 data.csv
# train_data.to_csv(os.path.join(config['data_path'], config['campaign_id'], 'data.csv'), index=False)
#
# # 绘制概率密度函数
# x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)
# pdf_values = normal_dist.pdf(x)
#
# plt.plot(x, pdf_values, label='PDF - Normal Distribution')
# plt.hist(train_data['value'], density=True, alpha=0.5, label='PDF - Original Data')
# plt.legend()
# plt.title('Probability Density Function')
# plt.show()
#
# # 绘制累积分布函数
# cdf_values = normal_dist.cdf(x)
#
# plt.plot(x, cdf_values, label='CDF - Normal Distribution')
# plt.hist(train_data['value'], density=True, cumulative=True, alpha=0.5, label='CDF - Original Data')
# plt.legend()
# plt.title('Cumulative Distribution Function')
# plt.show()



