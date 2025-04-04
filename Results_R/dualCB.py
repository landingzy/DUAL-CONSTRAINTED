import argparse
import os
import  time
import numpy as np
import pandas as pd
from TD3 import TD3
 
import warnings

warnings.filterwarnings("ignore")


def choose_init_base_bid(config, budget_para):  #resultF_roi_1_fixcpc
    base_bid_path = os.path.join('../lin_2/result/ipinyou/{}/normal/test'.format(config['campaign_id']),
                                 'test_bid_log.csv')
    if not os.path.exists(base_bid_path):
        raise FileNotFoundError('Run LIN first before you train drlb')
    data = pd.read_csv(base_bid_path)
    base_bid = data[data['budget_prop'] == budget_para].iloc[0]['base_bid']
    avg_pctr = data[data['budget_prop'] == budget_para].iloc[0]['average_pctr']

    return avg_pctr, base_bid


def cal_value(data, ecpc):
    data['pctr'] = data['pctr'].astype(float)

    data['value'] = np.clip(np.multiply(data['pctr'], ecpc).astype(int), 0, 300)

    return data


def get_budget(data):
    _ = []
    for day in data['day'].unique():
        current_day_budget = sum(data[data['day'].isin([day])]['market_price'])
        _.append(current_day_budget)

    return _


def bidding(pctr, para):
    bid = np.clip(np.multiply(pctr, para).astype(int), 0, 300)
    return bid


def bidding1(pctr, para):
    bid1 = np.clip(np.multiply(pctr, para).astype(int), 0, 300)
    return bid1


def reward_func(reward_type, lin_result, rl_result):
    fab_clks = rl_result['win_clks']
    hb_clks = lin_result['win_clks']
    fab_cost = rl_result['spend']
    hb_cost = lin_result['spend']
    fab_pctrs = rl_result['win_pctr']
    hb_pctrs = lin_result['win_pctr']
    hb_value = lin_result['value']
    fab_value = rl_result['value']
   

    if fab_clks >= hb_clks and fab_cost < hb_cost:
        r = 5
    elif fab_clks >= hb_clks and fab_cost >= hb_cost:
        r = 1
    elif fab_clks < hb_clks and fab_cost >= hb_cost:
        r = -5
    else:
        r = -2.5

   

    if reward_type == 'op':
        return r / 1000
    elif reward_type == 'nop':
        return r
    elif reward_type == 'nop_2.0':
        return fab_clks / 1000
    elif reward_type == 'pctr':
        return fab_pctrs
   

    else:
        return fab_clks


def bid(data, budget, **cfg):
    bid_imps = 0
    bid_clks = 0
    bid_pctr = 0
    win_imps = 0
    win_clks = 0
    win_pctr = 0
    slot_spend = 0

    bid_action = []
    value = 0
    spend = 0
    win_roi = []
    slot_action = []
    if len(data) == 0:
        return {
            'bid_imps': bid_imps,
            'bid_clks': bid_clks,
            'bid_pctr': bid_pctr,
            'win_imps': win_imps,
            'win_clks': win_clks,
            'win_pctr': win_pctr,
            'spend': spend,
            'bid_action': bid_action,
            'value': value,
            'win_roi': win_roi,
            'slot_action': slot_action
        }

    data['bid_price'] = np.clip(np.multiply(data['pctr'], cfg['slot_bid_para']).astype(int), 0, 300)
   

    data['win'] = data.apply(
     lambda x: 1 if x['bid_price'] >= x['market_price'] else 0, axis=1)
    

    win_data = data[data['win'] == 1]

    win_data['roi'] = np.divide(win_data['value'], win_data['market_price'])
    bid_action.extend(data.values.tolist())

    if len(win_data) == 0:
        return {
            'bid_imps': len(data),
            'bid_clks': sum(data['clk']),
            'bid_pctr': sum(data['pctr']),
            'win_imps': win_imps,
            'win_clks': win_clks,
            'win_pctr': win_pctr,
            'spend': spend,
            'bid_action': bid_action,
            'value': value,
            'win_roi': win_roi,
            'slot_action': slot_action
        }

    win_data['cumsum'] = win_data['market_price'].cumsum()

    if win_data.iloc[-1]['cumsum'] > budget:
        win_data = win_data[win_data['cumsum'] <= budget]

    bid_imps = len(data)
    bid_clks = sum(data['clk'])
    bid_pctr = sum(data['pctr'])
    win_imps = len(win_data)
    win_clks = sum(win_data['clk'])
    win_pctr = sum(win_data['pctr'])
    spend = sum(win_data['market_price'])
    value = sum(win_data['value'])

    slot_action.append([win_clks, win_imps, win_pctr])

    return {
        'bid_imps': bid_imps,
        'bid_clks': bid_clks,
        'bid_pctr': bid_pctr,
        'win_imps': win_imps,
        'win_clks': win_clks,
        'win_pctr': win_pctr,
        'spend': spend,
        'bid_action': bid_action,
        'value': value,
        'win_roi': win_roi,
        'slot_action': slot_action
    }


def rtb(data, budget_para, RL, config, train=True):
    if train:
        RL.is_test = False
    else:
        RL.is_test = True

    time_fraction = config['time_fraction']

    budget = get_budget(data)
    budget = np.divide(budget, budget_para)

    avg_pctr, base_bid = choose_init_base_bid(config, budget_para)
    ecpc = base_bid / avg_pctr
    data = cal_value(data, ecpc)

    episode_bid_imps = []
    episode_bid_clks = []
    episode_bid_pctr = []
    episode_win_imps = []
    episode_win_clks = []
    episode_win_pctr = []
    episode_spend = []
    episode_bid_action = []

    episode_action = []
    episode_reward = []
    episode_roi = []
    episode_roi_action = []
    # 记录训练过程中的 critic_loss
    critic_losses = []
    for day_index, day in enumerate(data['day'].unique()):
        day_bid_imps = []
        day_bid_clks = []
        day_bid_pctr = []
        day_win_imps = []
        day_win_clks = []
        day_win_pctr = []

        day_spend = []
        day_bid_action = []

        day_action = []
        day_reward = []

        day_data = data[data['day'].isin([day])]
        day_budget = [budget[day_index]]
        day_value = []
        day_roi = []
        day_roi_action = []
        for slot in range(0, time_fraction):
            slot_budget = budget[day_index] / time_fraction
            if slot == 0:
                state = [0, 1, 0, 0, 0, 0]
            else:
                left_slot_ratio = (time_fraction - 1 - slot) / (time_fraction - 1)

                state = [
                    slot,
                    (day_budget[-1] / day_budget[0]) / left_slot_ratio if left_slot_ratio else day_budget[-1] /
                                                                                               day_budget[0],
                    day_spend[-1] / day_budget[0],
                    day_win_clks[-1] / day_win_imps[-1] if day_win_imps[-1] else 0,
                    day_win_imps[-1] / day_bid_imps[-1] if day_bid_imps[-1] else 0,
                    day_value[-1] / day_spend[-1] if day_spend[-1] else 0,
                ]
            slot_data = day_data[day_data['time_fraction'] == slot]
            slot_action = RL.select_action(state)
            day_action.append(slot_action)

            slot_lin_bid_para = base_bid / avg_pctr

            slot_rl_bid_para = slot_lin_bid_para * slot_action
            # slot_rl_bid_para = slot_action

            slot_lin_result = bid(slot_data, day_budget[-1], slot_bid_para=slot_lin_bid_para)
            slot_rl_result = bid(slot_data, day_budget[-1], slot_bid_para=slot_rl_bid_para)

            slot_reward = reward_func(config['reward_type'], slot_lin_result, slot_rl_result)
            day_reward.append(slot_reward)

            p_slot = slot_rl_result['spend']
            v_slot = slot_rl_result['value']

            day_bid_imps.append(slot_rl_result['bid_imps'])
            day_bid_clks.append(slot_rl_result['bid_clks'])
            day_bid_pctr.append(slot_rl_result['bid_pctr'])
            day_win_imps.append(slot_rl_result['win_imps'])
            day_win_clks.append(slot_rl_result['win_clks'])
            day_win_pctr.append(slot_rl_result['win_pctr'])
            day_value.append(slot_rl_result['value'])
            day_spend.append(slot_rl_result['spend'])
            day_budget.append(day_budget[-1] - slot_rl_result['spend'])
            day_bid_action.extend(slot_rl_result['bid_action'])

            day_roi.append(slot_rl_result['slot_action'])
            day_roi_action.extend(slot_rl_result['win_roi'])
            if slot == time_fraction - 1:
                done = 1
                day_budget.pop(-1)
            else:
                done = 0

            left_slot_ratio = (time_fraction - 2 - slot) / (time_fraction - 1)
            next_state = [
                slot + 1,
                (day_budget[-1] / day_budget[0]) / left_slot_ratio if left_slot_ratio else day_budget[-1] /
                                                                                           day_budget[0],
                day_spend[-1] / day_budget[0],
                day_win_clks[-1] / day_win_imps[-1] if day_win_imps[-1] else 0,
                day_win_imps[-1] / day_bid_imps[-1] if day_bid_imps[-1] else 0,
                day_value[-1] / day_spend[-1] if day_spend[-1] else 0,
            ]

           

            if train:
                RL.store(slot_reward, next_state, done)
                if len(RL.memory) >= RL.batch_size and RL.total_step > RL.initial_random_steps:
                    actor_loss, critic_loss = RL.update_model()
                    global actor_loss_cnt
                    global critic_loss_cnt
                    actor_loss_cnt += 1
                    critic_loss_cnt += 1
                    RL.writer.add_scalar('actor_loss', actor_loss, actor_loss_cnt)
                    RL.writer.add_scalar('critic_loss', critic_loss, critic_loss_cnt)

            if done:
                break

        episode_bid_imps.append(sum(day_bid_imps))
        episode_bid_clks.append(sum(day_bid_clks))
        episode_bid_pctr.append(sum(day_bid_pctr))
        episode_win_imps.append(sum(day_win_imps))
        episode_win_clks.append(sum(day_win_clks))
        episode_win_pctr.append(sum(day_win_pctr))
        episode_spend.append(sum(day_spend))

        episode_bid_action.extend(day_bid_action)
        episode_roi_action.extend(day_roi_action)

        episode_roi.extend(day_roi)
        episode_action.append(day_action)
        episode_reward.append(sum(day_reward))

    if train:
        result = "训练"
    else:
        result = "测试"

    print(
        result + "：点击数 {}, 真实点击数 {}, pCTR {:.4f}, 真实pCTR {:.4f}, 赢标数 {}, 真实曝光数 {}, 花费 {}, CPM {:.4f}, CPC {:.4f}, 奖励 {:.2f}".format(
            int(sum(episode_win_clks)),
            int(sum(episode_bid_clks)),
            sum(episode_win_pctr),
            sum(episode_bid_pctr),
            sum(episode_win_imps),
            sum(episode_bid_imps),
            sum(episode_spend),
            sum(episode_spend) / sum(episode_win_imps) if sum(episode_win_imps) != 0 else 0,
            sum(episode_spend) / sum(episode_win_clks) if sum(episode_win_clks) != 0 else 0,
            sum(episode_reward)
        )
    )

    if train:
        global train_reward_cnt
        train_reward_cnt += 1
        RL.writer.add_scalar('train_reward', sum(episode_reward), train_reward_cnt)
    else:
        global test_reward_cnt
        test_reward_cnt += 1
        RL.writer.add_scalar('test_reward', sum(episode_reward), test_reward_cnt)

    episode_record = [
        int(sum(episode_win_clks)),
        int(sum(episode_bid_clks)),
        sum(episode_win_pctr),
        sum(episode_bid_pctr),
        sum(episode_win_imps),
        sum(episode_bid_imps),
        sum(episode_spend),
        sum(episode_spend) / sum(episode_win_imps) if sum(episode_win_imps) != 0 else 0,
        sum(episode_spend) / sum(episode_win_clks) if sum(episode_win_clks) != 0 else 0,
        sum(episode_reward)
    ]
    return episode_record, episode_action, episode_bid_action, episode_roi_action, episode_roi, critic_losses


def main(budget_para, RL, config):
    record_path = os.path.join(config['result_path'], config['campaign_id'])
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    # train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train' + str(budget_para) + '.csv'))
    # test_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'test' + str(budget_para) + '.csv'))

    train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.csv'))
    test_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'test.csv'))

    header = ['clk', 'pctr', 'value', 'market_price', 'day']

    if config['time_fraction'] == 96:
        header.append('96_time_fraction')
    elif config['time_fraction'] == 48:
        header.append('48_time_fraction')
    elif config['time_fraction'] == 24:
        header.append('24_time_fraction')

    train_data = train_data[header]
    train_data.columns = ['clk', 'pctr', 'value', 'market_price', 'day', 'time_fraction']
    test_data = test_data[header]
    test_data.columns = ['clk', 'pctr', 'value', 'market_price', 'day', 'time_fraction']

    epoch_train_record = []
    epoch_train_action = []

    epoch_test_record = []
    epoch_test_action = []
    train_epoch_critic_loss=[]
    test_epoch_critic_loss = []
    for epoch in range(config['train_epochs']):
        print('第{}轮'.format(epoch + 1))
        train_record, train_action, train_bid_action, train_roi_action, train_roi,train_loss = rtb(train_data, budget_para, RL,
                                                                                       config)
        
        test_record, test_action, test_bid_action, test_roi_action, test_roi,test_loss = rtb(test_data, budget_para, RL, config,
                                                                                 train=False)
         
        
        epoch_train_record.append(train_record)
        epoch_train_action.append(train_action)

        epoch_test_record.append(test_record)
        epoch_test_action.append(test_action)

        
       

        
        if config['save_loss']:
            loss_action_path = os.path.join(record_path, 'loss')
            if not os.path.exists(loss_action_path):
                os.makedirs(loss_action_path)
        loss_df = pd.DataFrame(data=train_loss)
        loss_df.to_csv( loss_action_path  + '/loss_' + str(budget_para)  + '.csv',
                           index=False)
                           
        if config['save_bid_action']:
            bid_action_path = os.path.join(record_path, 'bid_action')
            if not os.path.exists(bid_action_path):
                os.makedirs(bid_action_path)
        if config['save_roi_action']:
            roi_action_path = os.path.join(record_path, 'roi_action')
            if not os.path.exists(roi_action_path):
                os.makedirs(roi_action_path)
        if config['save_slot_roi_action']:
            roi_path = os.path.join(record_path, 'slot_roi_action')
            if not os.path.exists(roi_path):
                os.makedirs(roi_path)

        test_roi_action_df = pd.DataFrame(data=test_roi_action,
                                          columns=['clk', 'pctr', 'value', 'market_price', 'day', 'time_fraction',
                                                   'bid_price', 'win', 'roi'])
        test_roi_action_df.to_csv(roi_action_path + '/test_' + str(budget_para) + '_' + str(epoch) + '.csv',
                                  index=False)
        test_roi_df = pd.DataFrame(data=test_roi,
                                   columns=['test'])
        test_roi_df.to_csv(roi_path + '/test_' + str(budget_para) + '_' + str(epoch) + '.csv',
                           index=False)
        # train_bid_action_df = pd.DataFrame(data=train_bid_action,
        #                                    columns=['clk', 'pctr', 'market_price', 'day', 'time_fraction',
        #                                             'bid_price', 'win'])
        # train_bid_action_df.to_csv(bid_action_path + '/train_' + str(budget_para) + '_' + str(epoch) + '.csv',
        #                            index=False)

        # test_bid_action_df = pd.DataFrame(data=test_bid_action,
        #  columns=['clk', 'pctr','value',  'market_price', 'day', 'time_fraction',
        #      'bid_price', 'win'])
    # test_bid_action_df.to_csv(bid_action_path + '/test_' + str(budget_para) + '_' + str(epoch) + '.csv',
    # index=False)

    columns = ['clks', 'real_clks', 'pctr', 'real_pctr', 'imps', 'real_imps', 'spend', 'CPM', 'CPC', 'reward']

    train_record_df = pd.DataFrame(data=epoch_train_record, columns=columns)
    train_record_df.to_csv(record_path + '/train_episode_results_' + str(budget_para) + '.csv')

    train_action_df = pd.DataFrame(data=epoch_train_action)
    train_action_df.to_csv(record_path + '/train_episode_actions_' + str(budget_para) + '.csv')

    test_record_df = pd.DataFrame(data=epoch_test_record, columns=columns)
    test_record_df.to_csv(record_path + '/test_episode_results_' + str(budget_para) + '.csv')

    test_action_df = pd.DataFrame(data=epoch_test_action)
    test_action_df.to_csv(record_path + '/test_episode_actions_' + str(budget_para) + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ipinyou')
    parser.add_argument('--campaign_id', type=str, default='1458')
    parser.add_argument('--result_path', type=str, default='result_test1')
    parser.add_argument('--time_fraction', type=int, default=96)
    parser.add_argument('--feature_num', type=int, default=6)
    parser.add_argument('--action_num', type=int, default=1)
    parser.add_argument('--budget_para', nargs='+', default=[8, 16, 32])
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_bid_action', type=bool, default=False)
    parser.add_argument('--save_roi_action', type=bool, default=True)
    parser.add_argument('--save_loss', type=bool, default=True)

    parser.add_argument('--save_slot_roi_action', type=bool, default=True)
    parser.add_argument('--reward_type', type=str, default='op', help='op, nop_2.0, clk')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()
    config = vars(args)

    if not os.path.exists(config['result_path']):
        os.makedirs(config['result_path'])

    budget_para_list = list(map(int, config['budget_para']))

    actor_loss_cnt = 0
    critic_loss_cnt = 0
    train_reward_cnt = 0
    test_reward_cnt = 0

    for i in budget_para_list:
        RL = TD3(
            config['feature_num'],
            config['action_num'],
            config['memory_size'],
            config['batch_size'],
            initial_random_steps=200,
            # seed=config['seed']
        )
        print(type(RL.memory))

        print('当前预算条件{}'.format(i))
        main(i, RL, config)
