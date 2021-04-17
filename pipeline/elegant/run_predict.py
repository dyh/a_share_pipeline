import argparse
import sys

if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../ElegantRL_master')

from pipeline.elegant.env_predict import StockTradingEnvPredict

from pipeline.finrl import config

import os
import gym
import time
import torch
import numpy as np
import numpy.random as rd
import pandas as pd

from copy import deepcopy
from ElegantRL_master.elegantrl.agent import ReplayBuffer, ReplayBufferMP

gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

"""Plan to
Arguments(), let Arguments ask AgentXXX to get if_on_policy
Mega PPO and GaePPO into AgentPPO(..., if_gae)
"""

'''DEMO'''


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically

        '''Arguments for training (off-policy)'''
        self.net_dim = 2 ** 8  # the network width
        self.batch_size = 2 ** 8  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.target_step = 2 ** 10  # collect target_step, then update network
        self.max_memo = 2 ** 17  # capacity of replay buffer
        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9
            self.batch_size = 2 ** 8
            self.repeat_times = 2 ** 4
            self.target_step = 2 ** 12
            self.max_memo = self.target_step
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.show_gap = 2 ** 8  # show the Reward and Loss value per show_gap seconds
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main=True):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| There should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = str(self.gpu_id)
        if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
            self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.env.env_name}_{self.gpu_id}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


'''multiprocessing training'''

'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, show_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.show_gap = show_gap
        self.eva_times1 = eval_times1
        self.eva_times2 = eval_times2
        self.env = env
        self.target_reward = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def evaluate_save(self, act, steps, obj_a, obj_c) -> bool:
        reward_list = [get_episode_return(self.env, act, self.device)
                       for _ in range(self.eva_times1)]
        r_avg = np.average(reward_list)  # episode return average
        r_std = float(np.std(reward_list))  # episode return std

        if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
            reward_list += [get_episode_return(self.env, act, self.device)
                            for _ in range(self.eva_times2 - self.eva_times1)]
            r_avg = np.average(reward_list)  # episode return average
            r_std = float(np.std(reward_list))  # episode return std
        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            '''save actor.pth'''
            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")

        self.total_step += steps  # update total training steps
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

        if_reach_goal = bool(self.r_max > self.target_reward)  # check if_reach_goal
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_reward:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

        if time.time() - self.print_time > self.show_gap:
            self.print_time = time.time()
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
        return if_reach_goal

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''convert to array and save as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)


def get_episode_return(env, act, device) -> float:
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return env.episode_return if hasattr(env, 'episode_return') else episode_return


def save_learning_curve(recorder, cwd='.', save_title='learning curve'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_a = recorder[:, 3]
    obj_c = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    label = 'objA'
    ax11.set_ylabel(label, color=color11)
    ax11.plot(steps, obj_a, label=label, color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/plot_learning_curve.jpg")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def explore_before_training(env, buffer, target_step, reward_scale, gamma) -> int:
    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    steps = 0

    while steps < target_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        other = (scaled_reward, mask, action) if if_discrete else (scaled_reward, mask, *action)
        buffer.append_buffer(state, other)

        state = env.reset() if done else next_state
    return steps


#
# def get_npy(stock_code='sh.600036'):
#     parser = argparse.ArgumentParser(description='elegant')
#     # parser.add_argument('--multiprocess_id', type=str, default=multiprocess_id, help='multiprocess id')
#     parser.add_argument('--download_data', type=bool, default=True, help='download data')
#     args = parser.parse_args()
#
#     stock_dim = 1
#
#     # 创建目录
#     if not os.path.exists("./" + config.DATA_SAVE_DIR):
#         os.makedirs("./" + config.DATA_SAVE_DIR)
#     # if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
#     #     os.makedirs("./" + config.TRAINED_MODEL_DIR)
#     # if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
#     #     os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
#     # if not os.path.exists("./" + config.RESULTS_DIR):
#     #     os.makedirs("./" + config.RESULTS_DIR)
#     # if not os.path.exists("./" + config.LOGGER_DIR):
#     #     os.makedirs("./" + config.LOGGER_DIR)
#
#     # 股票代码
#     # stock_code = 'sh.600036'
#
#     # 训练开始日期
#     start_date = "2002-05-01"
#     # 停止训练日期 / 开始预测日期
#     # start_trade_date = "2021-03-08"
#     # 停止预测日期
#     end_date = '2021-04-14'
#
#     print("==============下载A股数据==============")
#     if args.download_data:
#         # 下载A股的日K线数据
#         stock_data = StockData(output_dir="./" + config.DATA_SAVE_DIR, date_start=start_date, date_end=end_date)
#         # 获得数据文件路径
#         csv_file_path = stock_data.download(stock_code, fields=stock_data.fields_day)
#     else:
#         csv_file_path = f"./{config.DATA_SAVE_DIR}/{stock_code}.csv"
#     pass
#
#     # csv_file_path = './datasets_temp/sh.600036.csv'
#     print("==============处理未来数据==============")
#
#     # open今日开盘价为T日数据，其余皆为T-1日数据，避免引入未来数据
#     df = pd.read_csv(csv_file_path)
#
#     # 删除未来数据，把df分为2个表，日期date+开盘open是A表，其余的是B表
#     df_left = df.drop(df.columns[2:], axis=1)
#
#     df_right = df.drop(['date', 'open'], axis=1)
#
#     # 删除A表第一行
#     df_left.drop(df_left.index[0], inplace=True)
#     df_left.reset_index(drop=True, inplace=True)
#
#     # 删除B表最后一行
#     df_right.drop(df_right.index[-1:], inplace=True)
#     df_right.reset_index(drop=True, inplace=True)
#
#     # 将A表和B表重新拼接，剔除了未来数据
#     df = pd.concat([df_left, df_right], axis=1)
#
#     # # 今天的数据，date、open为空，重新赋值
#     # df.loc[df.index[-1:], 'date'] = today_date
#     # df.loc[df.index[-1:], 'open'] = today_open_price
#
#     # 缓存文件，debug用
#     # df.to_csv(f'{config.DATA_SAVE_DIR}/{stock_code}_concat_df.csv', index=False)
#
#     print("==============加入技术指标==============")
#     fe = FeatureEngineer(
#         use_technical_indicator=True,
#         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
#         use_turbulence=False,
#         user_defined_feature=False,
#     )
#
#     df_fe = fe.preprocess_data(df)
#     df_fe['log_volume'] = np.log(df_fe.volume * df_fe.close)
#     df_fe['change'] = (df_fe.close - df_fe.open) / df_fe.close
#     df_fe['daily_variance'] = (df_fe.high - df_fe.low) / df_fe.close
#
#     # df_fe.to_csv(f'{config.DATA_SAVE_DIR}/{stock_code}_processed_df.csv', index=False)
#
#     print("==============拆分 训练/预测 数据集==============")
#     # Training & Trading data split
#     # df_train = df_fe[(df_fe.date >= start_date) & (df_fe.date < start_trade_date)]
#     # df_train = df_train.sort_values(["date", "tic"], ignore_index=True)
#     # df_train.index = df_train.date.factorize()[0]
#     #
#     # df_predict = df_fe[(df_fe.date >= start_trade_date) & (df_fe.date <= end_date)]
#     # df_predict = df_predict.sort_values(["date", "tic"], ignore_index=True)
#     # df_predict.index = df_predict.date.factorize()[0]
#
#     print("==============数据准备完成==============")
#
#     # data = pd.read_csv('./AAPL_processed.csv', index_col=0)
#
#     # from preprocessing.preprocessors import pd, data_split, preprocess_data, add_turbulence
#     #
#     # # the following is same as part of run_model()
#     # preprocessed_path = "done_data.csv"
#     # if if_load and os.path.exists(preprocessed_path):
#     #     data = pd.read_csv(preprocessed_path, index_col=0)
#     # else:
#     #     data = preprocess_data()
#     #     data = add_turbulence(data)
#     #     data.to_csv(preprocessed_path)
#
#     # df = df_fe
#
#     # print('df_fe.shape:', df_fe.shape)
#
#     df_fe.to_csv(f'{config.DATA_SAVE_DIR}/{stock_code}_processed.csv', index=False)
#
#     train__df = df_fe
#
#     print('train__df.shape:', train__df.shape)
#
#     # print(train__df) # df: DataFrame of Pandas
#
#     train_ary = train__df.to_numpy().reshape((-1, stock_dim, 25))
#
#     '''state_dim = 1 + 6 * stock_dim, stock_dim=30
#     n   item    index
#     1   ACCOUNT -
#     30  adjcp   2
#     30  stock   -
#     30  macd    7
#     30  rsi     8
#     30  cci     9
#     30  adx     10
#     '''
#     data_ary = np.empty((train_ary.shape[0], 5, stock_dim), dtype=np.float32)
#     # data_ary[:, 0] = train_ary[:, :, 4]  # adjcp
#     # data_ary[:, 1] = train_ary[:, :, 8]  # macd
#     # data_ary[:, 2] = train_ary[:, :, 11]  # rsi
#     # data_ary[:, 3] = train_ary[:, :, 12]  # cci
#     # data_ary[:, 4] = train_ary[:, :, 13]  # adx
#
#     data_ary[:, 0] = train_ary[:, :, 1]  # T open
#     data_ary[:, 1] = train_ary[:, :, 2]  # T-1 high
#     data_ary[:, 2] = train_ary[:, :, 3]  # T-1 low
#     data_ary[:, 3] = train_ary[:, :, 4]  # T-1 close
#     data_ary[:, 4] = train_ary[:, :, 14]  # T-1 macs
#
#     # 变形
#     data_ary = data_ary.reshape((-1, 5 * stock_dim))
#
#     # os.makedirs(data_path[:data_path.rfind('/')])
#     data_path = f"./{config.DATA_SAVE_DIR}/{stock_code}.npy"
#     np.save(data_path, data_ary.astype(np.float16))  # save as float16 (0.5 MB), float32 (1.0 MB)
#
#     print('data_ary.shape:', data_ary.shape)
#
#     print('| FinanceStockEnv(): save in:', data_path)
#     # return data_ary
#     pass
#

def demo3_custom_env_fin_rl(stock_code='sh.600036'):
    from ElegantRL_master.elegantrl.agent import AgentPPO

    '''choose an DRL algorithm'''
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.agent.if_use_gae = False

    # 训练开始日期
    start_date = '2021-03-08'
    # 停止训练日期 / 开始预测日期
    # start_trade_date = "2021-03-08"
    # 停止预测日期
    end_date = '2021-04-16'

    # data_path = f"./{config.DATA_SAVE_DIR}/{stock_code}.npy"

    # beg_i=0, end_i=3220, initial_amount=1e6, initial_stocks=None,
    # max_stock=1e2, buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99,
    # ticker_list=None, tech_id_list=None, beg_date=None, end_date=None,

    # train_env_kwargs = {
    #     "ticker_list": ['sh.600036'],
    #     "beg_date": start_date,
    #     "end_date": end_date,
    #     "beg_i": 1,  # 2002-05-10
    #     # "end_i": 4500,  # 2021-03-08
    #     "end_i": 4466,  # 2021-01-12
    #     "max_stock": 10000,
    #     "initial_amount": 100000,
    #     "buy_cost_pct": 0.0003,
    #     "sell_cost_pct": 0.0003,
    #     # "state_space": 1 + (2 + len(config.TECHNICAL_INDICATORS_LIST)) * stock_dimension,
    #     # "stock_dim": 1,
    #     "tech_id_list": config.TECHNICAL_INDICATORS_LIST,
    #     # "action_space": stock_dimension,
    #     # "reward_scaling": 1e-4
    # }

    eval_env_kwargs = {
        "ticker_list": ['sh.600036'],
        "beg_date": start_date,
        "end_date": end_date,
        # "beg_i": 4501,  # 2021-03-09
        "beg_i": 0,  # 2021-01-13
        "end_i": 28,  # 2021-04-14
        "max_stock": 4000,
        "initial_amount": 50000,
        "buy_cost_pct": 0.0003,
        "sell_cost_pct": 0.0003,
        # "state_space": 1 + (2 + len(config.TECHNICAL_INDICATORS_LIST)) * stock_dimension,
        # "stock_dim": 1,
        "tech_id_list": config.TECHNICAL_INDICATORS_LIST,
        # "action_space": stock_dimension,
        # "reward_scaling": 1e-4
    }

    args.env = StockTradingEnvPredict(**eval_env_kwargs)

    args.env_eval = StockTradingEnvPredict(**eval_env_kwargs)

    args.reward_scale = 2 ** 0  # RewardRange: 0 < 1.0 < 1.25 <
    args.break_step = int(5e6)
    args.net_dim = 2 ** 8
    args.max_step = args.env_eval.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.eval_times1 = 2 ** 2
    args.eval_times2 = 2 ** 4
    args.if_allow_break = True
    "TotalStep:  5e4, TargetReward: 1.25, UsedTime:  20s"
    "TotalStep: 20e4, TargetReward: 1.50, UsedTime:  80s"

    '''predict'''
    predict(args)


'''single process training'''


def predict(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    # env = args.env
    agent = args.agent
    # gpu_id = args.gpu_id  # necessary for Evaluator?
    env_eval = args.env_eval

    '''training arguments'''
    net_dim = args.net_dim
    # max_memo = args.max_memo
    # break_step = args.break_step
    # batch_size = args.batch_size
    # target_step = args.target_step
    # repeat_times = args.repeat_times
    # if_break_early = args.if_allow_break
    # gamma = args.gamma
    # reward_scale = args.reward_scale

    '''evaluating arguments'''
    # show_gap = args.show_gap
    # eval_times1 = args.eval_times1
    # eval_times2 = args.eval_times2
    # env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    env_eval = deepcopy(env_eval)
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env_eval.max_step
    state_dim = env_eval.state_dim
    action_dim = env_eval.action_dim
    if_discrete = env_eval.if_discrete
    # env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    env_eval = deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim)

    agent.save_load_model('./AgentPPO/', if_save=False)

    # if_on_policy = getattr(agent, 'if_on_policy', False)

    # buffer = ReplayBuffer(max_len=max_memo + max_step, if_on_policy=if_on_policy, if_gpu=True,
    #                       state_dim=state_dim, action_dim=1 if if_discrete else action_dim)
    #
    # evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
    #                       eval_times1=eval_times1, eval_times2=eval_times2, show_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    agent.state = env_eval.reset()
    # if if_on_policy:
    #     steps = 0
    # else:  # explore_before_training for off-policy
    #     with torch.no_grad():  # update replay buffer
    #         steps = explore_before_training(env, buffer, target_step, reward_scale, gamma)
    #
    #     agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update
    #     agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
    #     agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    # total_step = steps

    '''start training'''
    # if_reach_goal = False
    # while not ((if_break_early and if_reach_goal)
    #            or total_step > break_step
    #            or os.path.exists(f'{cwd}/stop')):
    #     with torch.no_grad():  # speed up running
    #         steps = agent.explore_env(env_eval, buffer, target_step, reward_scale, gamma)
    #
    #     total_step += steps
    #
    #     obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)
    #
    #     with torch.no_grad():  # speed up running
    #         if_reach_goal = evaluator.evaluate_save(agent.act, steps, obj_a, obj_c)

    # get_episode_return(self.env, act, self.device)

    episode_return = 0.0  # sum of rewards in an episode
    # max_step = env_eval.max_step
    if_discrete = env_eval.if_discrete

    state = env_eval.reset()
    # for _ in range(max_step):

    with torch.no_grad():  # speed up running
        while True:
            s_tensor = torch.as_tensor((state,), device=agent.device)
            a_tensor = agent.act(s_tensor)
            if if_discrete:
                a_tensor = a_tensor.argmax(dim=1)
            action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
            state, reward, done, _ = env_eval.step(action)

            episode_return += reward
            if done:
                break
        pass


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # get_npy(stock_code='sh.600036')

    demo3_custom_env_fin_rl(stock_code='sh.600036')
