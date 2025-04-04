import os.path

import config
from model import RLB_DP_I
from utils import *

# 配置参数
obj_type = "clk"
clk_vp = 1
N = 1000
c0 = 1 / 2
gamma = 1

# 源路径
src = "ipinyou"

# 日志
performance_dir = 'bid-performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)
log_in = open(
    config.projectPath + "bid-performance/{}_N={}_c0={}_obj={}_clkvp={}.txt".format(src, N, c0, obj_type, clk_vp), "w")
print("logs in {}".format(log_in.name))
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}\t {:>8}" \
    .format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC", "win-pctr")
print(log)
log_in.write(log + "\n")

# 配置参数
camps = config.ipinyou_camps
data_path = config.ipinyouPath
max_market_price = config.ipinyou_max_market_price

# 训练



for camp in camps:
    camp_info = config.get_camp_info(camp)  # 加载广告活动静态信息
    auction_in = open(data_path + camp + "/value_test.bid.rlb.txt", "r")  # 读入数据集
    opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))  # 生成点击对象
    B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)  # 生成预算

    m_pdf = calc_m_pdf(camp_info["price_counter_train"])  # 计算每个价格下展示机会的概率
    rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)

    bid_model_dir = 'bid-model/{}'.format(camp)
    if not os.path.exists(bid_model_dir):
        os.makedirs(bid_model_dir)
    rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
                                                              os.path.join(bid_model_dir, "v_nb_N={}.txt".format(N)))

    # RLB
    rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
    setting = "{}, camp={}, algo={}, N={}, c0={}" \
        .format(src, camp, "rlb", N, c0)
    if not os.path.exists(os.path.join(config.projectPath, "bid-log")):
        os.makedirs(os.path.join(config.projectPath, "bid-log"))
    bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

    # model_path = config.projectPath + "/bid-model/v_nb_N={}.txt".format(N)
    model_path = os.path.join(bid_model_dir, "v_nb_N={}.txt".format(N))
    rlb_dp_i.load_value_function(N, B, model_path)

    (auction, imp, clk, cost, pctr) = rlb_dp_i.run(auction_in, bid_log_path, N, c0,
                                             max_market_price, delimiter=" ", save_log=True)

    win_rate = imp / auction * 100
    cpm = (cost / 1000) / imp * 1000
    ecpc = (cost / 1000) / clk
    obj = opt_obj.get_obj(imp, clk, cost)
    log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}\t {:>8.4f}" \
        .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc, pctr)
    print(log)
    log_in.write(log + "\n")

log_in.flush()
log_in.close()
