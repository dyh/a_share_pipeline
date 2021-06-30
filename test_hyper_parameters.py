import os
import random
import shutil
import time
from datetime import datetime

import config
import numpy as np
from stock_data import StockData

from agent_single import AgentPPO, AgentSAC, AgentTD3, AgentDDPG, AgentModSAC, AgentDuelingDQN, AgentSharedSAC, \
    AgentDoubleDQN

from run_single import Arguments, train_and_evaluate_mp, train_and_evaluate

from utils.date_time import get_datetime_from_date_str, time_point, get_begin_vali_date_list
from env_train_single import StockTradingEnv
from env_predict_single import StockTradingEnvPredict

if __name__ == '__main__':

    np.set_printoptions(suppress=True)

    # 开始训练的日期，在程序启动之后，不要改变。
    config.SINGLE_A_STOCK_CODE = ['sh.600036', ]

    # 初始现金
    initial_capital = 150000

    # 单次 购买/卖出 最大股数
    max_stock = 3000

    initial_stocks_train = np.zeros(len(config.SINGLE_A_STOCK_CODE), dtype=np.float32)
    initial_stocks_vali = np.zeros(len(config.SINGLE_A_STOCK_CODE), dtype=np.float32)

    # 默认持有1000股
    initial_stocks_train[0] = 1000.0
    initial_stocks_vali[0] = 1000.0

    # break_step = 50000
    # break_step = int(1e6)
    # break_step = int(3e6)

    if_on_policy = False
    # if_use_gae = True

    config.IF_SHOW_PREDICT_INFO = False

    config.START_DATE = "2002-05-01"
    config.START_EVAL_DATE = "2019-04-26"
    config.END_DATE = "2021-05-21"

    # 更新股票数据
    StockData.update_stock_data(tic_code=config.SINGLE_A_STOCK_CODE[0])

    # 好用 AgentPPO(), # AgentSAC(), AgentTD3(), AgentDDPG(), AgentModSAC(),
    # AgentDoubleDQN 单进程好用?
    # 不好用 AgentDuelingDQN(), AgentDoubleDQN(), AgentSharedSAC()
    # 选择agent
    for agent_item in ['AgentDDPG', 'AgentPPO', 'AgentSAC', 'AgentTD3', ]:

        config.AGENT_NAME = agent_item

        work_days = 518

        # 开始的时间
        time_begin = datetime.now()

        # 更新工作日标记，用于 run_single.py 加载训练过的 weights 文件
        config.VALI_DAYS_FLAG = str(work_days)

        model_folder_path = f'./{config.WEIGHTS_PATH}/single/{config.AGENT_NAME}/{config.SINGLE_A_STOCK_CODE[0]}' \
                            f'/single_{config.VALI_DAYS_FLAG}'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            pass

        print('\r\n')
        print('-' * 40)
        print('config.AGENT_NAME', config.AGENT_NAME)
        print('# 训练-预测周期', config.START_DATE, '-', config.START_EVAL_DATE, '-', config.END_DATE)
        print('# work_days', work_days)
        print('# model_folder_path', model_folder_path)
        print('# initial_capital', initial_capital)
        print('# max_stock', max_stock)

        # Agent
        # args = Arguments(if_on_policy=True)
        # off-policy DRL algorithm: AgentSAC, AgentTD3, AgentDDPG, AgentModSAC
        # on-policy DRL algorithm: AgentPPO
        agent_class = None
        if config.AGENT_NAME == 'AgentPPO':
            agent_class = AgentPPO()
            if_on_policy = True
            break_step = int(1e6)
            pass
        elif config.AGENT_NAME == 'AgentSAC':
            agent_class = AgentSAC()
            if_on_policy = False
            break_step = 10000
            pass
        elif config.AGENT_NAME == 'AgentTD3':
            agent_class = AgentTD3()
            if_on_policy = False
            break_step = 10000
            pass
        elif config.AGENT_NAME == 'AgentDDPG':
            agent_class = AgentDDPG()
            if_on_policy = False
            break_step = 10000
            pass
        elif config.AGENT_NAME == 'AgentModSAC':
            agent_class = AgentModSAC()
            if_on_policy = False
            break_step = 10000
            pass
        elif config.AGENT_NAME == 'AgentDuelingDQN':
            agent_class = AgentDuelingDQN()
            if_on_policy = False
            break_step = 10000
            pass
        elif config.AGENT_NAME == 'AgentSharedSAC':
            agent_class = AgentSharedSAC()
            if_on_policy = False
            break_step = 10000
            pass
        elif config.AGENT_NAME == 'AgentDoubleDQN':
            agent_class = AgentDoubleDQN()
            if_on_policy = False
            break_step = 10000
            pass
        else:
            break_step = 10000

        print('break_step', break_step)

        args = Arguments(if_on_policy=if_on_policy)
        args.agent = agent_class
        # args.agent.if_use_gae = if_use_gae
        args.agent.lambda_entropy = 0.04
        args.gpu_id = 0

        tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
            'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST

        gamma = 0.99
        # gamma = 2
        # gamma = random.uniform(0.99, 1.99)

        print('# initial_stocks_train', initial_stocks_train)
        print('# initial_stocks_vali', initial_stocks_vali)

        buy_cost_pct = 0.003
        sell_cost_pct = 0.003
        start_date = config.START_DATE
        start_eval_date = config.START_EVAL_DATE
        end_eval_date = config.END_DATE

        # train
        args.env = StockTradingEnv(cwd='', gamma=gamma, max_stock=max_stock,
                                   initial_capital=initial_capital,
                                   buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct, start_date=start_date,
                                   end_date=start_eval_date, env_eval_date=end_eval_date,
                                   ticker_list=config.SINGLE_A_STOCK_CODE,
                                   tech_indicator_list=tech_indicator_list, initial_stocks=initial_stocks_train,
                                   if_eval=False)

        # eval
        args.env_eval = StockTradingEnvPredict(cwd='', gamma=gamma, max_stock=max_stock,
                                               initial_capital=initial_capital,
                                               buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct,
                                               start_date=start_date,
                                               end_date=start_eval_date, env_eval_date=end_eval_date,
                                               ticker_list=config.SINGLE_A_STOCK_CODE,
                                               tech_indicator_list=tech_indicator_list,
                                               initial_stocks=initial_stocks_vali,
                                               if_eval=True)

        # args.env.target_reward = 10
        # args.env_eval.target_reward = 10

        args.env.target_return = 5
        args.env_eval.target_return = 5

        # Hyperparameters
        args.gamma = gamma

        print('gamma:', args.gamma)

        # args.reward_scale = 2 ** 10

        print('reward_scale:', args.reward_scale)

        # ----
        # args.break_step = int(5e6)
        # args.break_step = int(2e6)
        args.break_step = break_step
        # ----

        args.net_dim = 2 ** 9
        args.max_step = args.env.max_step

        # ----
        # args.max_memo = args.max_step * 4
        args.max_memo = (args.max_step - 1) * 8
        # ----

        # ----
        args.batch_size = 2 ** 11
        # args.batch_size = 2305
        # args.batch_size = random.randint(2**9, 2**13)
        print('batch_size:', args.batch_size)

        # ----

        # ----
        args.repeat_times = 2 ** 4
        # args.repeat_times = random.randint(2 ** 2, 2 ** 6)
        print('repeat_times:', args.repeat_times)
        # ----

        args.eval_gap = 2 ** 4
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        # args.eval_gap = random.randint(2 ** 2, 2 ** 6)
        # args.eval_times1 = random.randint(2 ** 2, 2 ** 6)
        # args.eval_times2 = random.randint(2 ** 2, 2 ** 6)

        print('eval_gap:', args.eval_gap)
        print('eval_times1:', args.eval_times1)
        print('eval_times2:', args.eval_times2)

        # ----
        # args.if_allow_break = False
        args.if_allow_break = True
        # ----

        args.rollout_num = 2  # the number of rollout workers (larger is not always faster)

        # train_and_evaluate(args)
        train_and_evaluate_mp(args)  # the training process will terminate once it reaches the target reward.

        # 保存训练后的模型
        shutil.copyfile(f'./{config.WEIGHTS_PATH}/StockTradingEnv-v1/actor.pth', f'{model_folder_path}/actor.pth')

        model_folder_path = f'./{config.WEIGHTS_PATH}/single/{config.AGENT_NAME}/{config.SINGLE_A_STOCK_CODE[0]}' \
                            f'/single_{config.VALI_DAYS_FLAG}'

        # 保存训练曲线图
        # plot_learning_curve.jpg
        timepoint_temp = time_point()
        plot_learning_curve_file_path = f'{model_folder_path}/plot_{timepoint_temp}.jpg'
        shutil.copyfile(f'./{config.WEIGHTS_PATH}/StockTradingEnv-v1/plot_learning_curve.jpg',
                        plot_learning_curve_file_path)

        # 结束预测的时间
        time_end = datetime.now()
        duration = (time_end - time_begin).seconds
        print('>', config.AGENT_NAME, break_step, 'steps', '训练耗时', duration, '秒')

        # sleep 60 秒
        print('sleep 20 秒')
        time.sleep(20)
        pass
    pass
