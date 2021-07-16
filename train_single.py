import os
import shutil
import time
from datetime import datetime

import config
import numpy as np
from stock_data import StockData

from agent_single import AgentPPO, AgentSAC, AgentTD3, AgentDDPG, AgentModSAC, AgentDuelingDQN, AgentSharedSAC, \
    AgentDoubleDQN

from run_single import Arguments, train_and_evaluate_mp, train_and_evaluate
from train_helper import init_model_hyper_parameters_table_sqlite, query_model_hyper_parameters_sqlite, \
    update_model_hyper_parameters_by_reward_history, clear_train_history_table_sqlite
from utils import date_time

from utils.date_time import get_datetime_from_date_str, get_next_work_day, \
    get_today_date
from env_train_single import StockTradingEnv
from env_predict_single import StockTradingEnvPredict

if __name__ == '__main__':

    # 初始化超参表
    init_model_hyper_parameters_table_sqlite()

    predict_work_days = 100

    # 开始训练的日期，在程序启动之后，不要改变
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

    if_on_policy = False
    # if_use_gae = True

    config.IF_SHOW_PREDICT_INFO = False

    config.START_DATE = "2003-05-01"
    config.START_EVAL_DATE = ""

    # 整体结束日期，今天的日期，减去90工作日
    # config.END_DATE = str(get_next_work_day(get_datetime_from_date_str(get_today_date()), -predict_work_days))
    config.END_DATE = '2021-07-14'

    # 更新股票数据
    StockData.update_stock_data(tic_code=config.SINGLE_A_STOCK_CODE[0])

    # 好用 AgentPPO(), # AgentSAC(), AgentTD3(), AgentDDPG(), AgentModSAC(),
    # AgentDoubleDQN 单进程好用?
    # 不好用 AgentDuelingDQN(), AgentDoubleDQN(), AgentSharedSAC()

    loop_index = 0

    # 循环
    while True:

        # 清空训练历史记录表
        clear_train_history_table_sqlite()

        # 从 model_hyper_parameters 表中，找到 training_times 最小的记录
        # 获取超参
        hyper_parameters_id, hyper_parameters_model_name, if_on_policy, break_step, train_reward_scale, \
        eval_reward_scale, training_times, time_point \
            = query_model_hyper_parameters_sqlite()

        if if_on_policy == 'True':
            if_on_policy = True
        else:
            if_on_policy = False
        pass

        config.MODEL_HYPER_PARAMETERS = str(hyper_parameters_id)

        # 获得Agent参数
        agent_class = None
        train_reward_scaling = 2 ** train_reward_scale
        eval_reward_scaling = 2 ** eval_reward_scale

        # 模型名称
        config.AGENT_NAME = str(hyper_parameters_model_name).split('_')[0]

        if config.AGENT_NAME == 'AgentPPO':
            agent_class = AgentPPO()
        elif config.AGENT_NAME == 'AgentSAC':
            agent_class = AgentSAC()
        elif config.AGENT_NAME == 'AgentTD3':
            agent_class = AgentTD3()
        elif config.AGENT_NAME == 'AgentDDPG':
            agent_class = AgentDDPG()
        elif config.AGENT_NAME == 'AgentModSAC':
            agent_class = AgentModSAC()
        elif config.AGENT_NAME == 'AgentDuelingDQN':
            agent_class = AgentDuelingDQN()
        elif config.AGENT_NAME == 'AgentSharedSAC':
            agent_class = AgentSharedSAC()
        elif config.AGENT_NAME == 'AgentDoubleDQN':
            agent_class = AgentDoubleDQN()
        pass

        # 预测周期
        work_days = int(str(hyper_parameters_model_name).split('_')[1])

        # 预测的截止日期
        end_vali_date = get_datetime_from_date_str(config.END_DATE)

        # 开始预测日期
        begin_date = date_time.get_next_work_day(end_vali_date, next_flag=-work_days)

        # 统计耗时
        time_begin = datetime.now()

        # 更新工作日标记，用于 run_single.py 加载训练过的 weights 文件
        config.VALI_DAYS_FLAG = str(work_days)

        model_folder_path = f'./{config.WEIGHTS_PATH}/single/{config.AGENT_NAME}/{config.SINGLE_A_STOCK_CODE[0]}' \
                            f'/single_{config.VALI_DAYS_FLAG}'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            pass

        # 开始预测的日期
        config.START_EVAL_DATE = str(begin_date)

        print('\r\n')
        print('-' * 40)
        print('config.AGENT_NAME', config.AGENT_NAME)
        print('# 训练-预测周期', config.START_DATE, '-', config.START_EVAL_DATE, '-', config.END_DATE)
        print('# work_days', work_days)
        print('# model_folder_path', model_folder_path)
        print('# initial_capital', initial_capital)
        print('# max_stock', max_stock)

        args = Arguments(if_on_policy=if_on_policy)
        args.agent = agent_class
        # args.agent.if_use_gae = if_use_gae
        args.agent.lambda_entropy = 0.04
        args.gpu_id = 0

        tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
            'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST

        gamma = 0.99

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

        args.env.target_return = 10
        args.env_eval.target_return = 10

        # 奖励 比例
        args.env.reward_scaling = train_reward_scaling
        args.env_eval.reward_scaling = eval_reward_scaling

        print('train reward_scaling', args.env.reward_scaling)
        print('eval reward_scaling', args.env_eval.reward_scaling)

        # Hyperparameters
        args.gamma = gamma
        # args.gamma = 0.99

        # reward_scaling 在 args.env里调整了，这里不动
        args.reward_scale = 2 ** 0

        # TODO ----
        args.break_step = int(break_step / 5)
        # args.break_step = 1200

        print('break_step', args.break_step)

        args.net_dim = 2 ** 9
        args.max_step = args.env.max_step

        # args.max_memo = args.max_step * 4
        args.max_memo = (args.max_step - 1) * 8

        args.batch_size = 2 ** 12
        # args.batch_size = 2305
        print('batch_size', args.batch_size)

        # ----
        # args.repeat_times = 2 ** 3
        args.repeat_times = 2 ** 4
        # ----

        args.eval_gap = 2 ** 4
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        args.if_allow_break = False

        args.rollout_num = 2  # the number of rollout workers (larger is not always faster)

        # train_and_evaluate(args)
        train_and_evaluate_mp(args)  # the training process will terminate once it reaches the target reward.

        # 保存训练后的模型
        shutil.copyfile(f'./{config.WEIGHTS_PATH}/StockTradingEnv-v1/actor.pth', f'{model_folder_path}/actor.pth')

        model_folder_path = f'./{config.WEIGHTS_PATH}/single/{config.AGENT_NAME}/{config.SINGLE_A_STOCK_CODE[0]}' \
                            f'/single_{config.VALI_DAYS_FLAG}'

        # 保存训练曲线图
        # plot_learning_curve.jpg
        timepoint_temp = date_time.time_point()
        plot_learning_curve_file_path = f'{model_folder_path}/plot_{timepoint_temp}.jpg'
        shutil.copyfile(f'./{config.WEIGHTS_PATH}/StockTradingEnv-v1/plot_learning_curve.jpg',
                        plot_learning_curve_file_path)

        # 训练结束后，model_hyper_parameters 表 中的 训练的次数 +1，训练的时间点 更新。

        # 判断 train_history 表，是否有记录，如果有，则整除 256 + 128。将此值更新到 model_hyper_parameters 表的 超参，减去相应的值。

        update_model_hyper_parameters_by_reward_history(model_hyper_parameters_id=hyper_parameters_id,
                                                        origin_train_reward_scale=train_reward_scale,
                                                        origin_eval_reward_scale=eval_reward_scale,
                                                        origin_training_times=training_times)

        # 结束预测的时间
        time_end = datetime.now()
        duration = (time_end - time_begin).seconds

        print('>', config.AGENT_NAME, break_step, 'steps', '训练耗时', duration, '秒')

        # 循环次数
        loop_index += 1
        print('>', 'while 循环次数', loop_index, '\r\n')

        print('sleep 10 秒\r\n')
        time.sleep(10)

        pass
