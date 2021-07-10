import os
import shutil
import time
from datetime import datetime

import config
import numpy as np
from stock_data import StockData

from agent_single import AgentPPO, AgentSAC, AgentTD3, AgentDDPG, AgentModSAC, AgentDuelingDQN, AgentSharedSAC, \
    AgentDoubleDQN

from run_single import Arguments, train_and_evaluate_mp

from utils.date_time import get_datetime_from_date_str, time_point, get_begin_vali_date_list, get_next_work_day, \
    get_today_date
from env_train_single import StockTradingEnv
from env_predict_single import StockTradingEnvPredict


# 定向训练
def filter_date(list_begin_vali_date_temp):
    # 获取7个日期list
    list_filter_date = []

    # 过滤日期，专项预测
    if config.AGENT_NAME == 'AgentPPO':
        list_filter_date = [50, 100, 150, 200, 300, 500, 1000]
        pass
    elif config.AGENT_NAME == 'AgentSAC':
        list_filter_date = [50, 100, 150, 200, 300, 500, 1000]
        pass
    elif config.AGENT_NAME == 'AgentTD3':
        list_filter_date = [50, 100, 150, 200, 300, 500, 1000]
        pass
    elif config.AGENT_NAME == 'AgentDDPG':
        list_filter_date = [50, 100, 150, 200, 300, 500, 1000]
        pass
    elif config.AGENT_NAME == 'AgentModSAC':
        list_filter_date = [50, 100, 150, 200, 300, 500, 1000]
        pass
    elif config.AGENT_NAME == 'AgentDuelingDQN':
        pass
    elif config.AGENT_NAME == 'AgentSharedSAC':
        pass
    elif config.AGENT_NAME == 'AgentDoubleDQN':
        pass
    pass

    list_result = []
    for work_days1, begin_date1 in list_begin_vali_date_temp:
        if work_days1 in list_filter_date:
            list_result.append((work_days1, begin_date1))
        pass
    pass

    return list_result
    pass


# 获得agent参数
def get_agent_args():
    agent_class1 = None
    if_on_policy1 = None
    break_step1 = 0
    train_reward_scaling1 = 0
    eval_reward_scaling1 = 0

    if config.AGENT_NAME == 'AgentPPO':
        agent_class1 = AgentPPO()
        if_on_policy1 = True
        break_step1 = int(1e6)

        if 20 <= int(config.VALI_DAYS_FLAG) < 30:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -6
            pass
        elif 30 <= int(config.VALI_DAYS_FLAG) < 40:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -6
            pass
        elif 40 <= int(config.VALI_DAYS_FLAG) < 50:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -8
            pass
        elif 50 <= int(config.VALI_DAYS_FLAG) < 60:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -8
            pass
        elif 60 <= int(config.VALI_DAYS_FLAG) < 90:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -8
            pass
        elif 90 <= int(config.VALI_DAYS_FLAG) < 300:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -8
            pass
        elif 300 <= int(config.VALI_DAYS_FLAG) < 500:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -19
            pass
        elif 500 <= int(config.VALI_DAYS_FLAG):
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -10
            pass
        else:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -8
            pass
        pass
    elif config.AGENT_NAME == 'AgentSAC':
        agent_class1 = AgentSAC()
        if_on_policy1 = False
        break_step1 = 10000

        if int(config.VALI_DAYS_FLAG) < 50:
            train_reward_scaling1 = 2 ** -8
            eval_reward_scaling1 = 2 ** -7
            pass
        elif 50 <= int(config.VALI_DAYS_FLAG) < 60:
            train_reward_scaling1 = 2 ** -9
            eval_reward_scaling1 = 2 ** -7
            pass
        elif 60 <= int(config.VALI_DAYS_FLAG) < 90:
            train_reward_scaling1 = 2 ** -10
            eval_reward_scaling1 = 2 ** -7
            pass
        elif 90 <= int(config.VALI_DAYS_FLAG) < 1000:
            train_reward_scaling1 = 2 ** -9
            eval_reward_scaling1 = 2 ** -8
            pass
        elif 1000 <= int(config.VALI_DAYS_FLAG) < 1268:
            train_reward_scaling1 = 2 ** -10
            eval_reward_scaling1 = 2 ** -9
            pass
        elif 1268 <= int(config.VALI_DAYS_FLAG):
            train_reward_scaling1 = 2 ** -10
            eval_reward_scaling1 = 2 ** -10
            pass
        pass

    elif config.AGENT_NAME == 'AgentTD3':
        agent_class1 = AgentTD3()
        if_on_policy1 = False
        break_step1 = 50000

        if 20 <= int(config.VALI_DAYS_FLAG) < 30:
            train_reward_scaling1 = 2 ** -6
            eval_reward_scaling1 = 2 ** -5
            pass
        elif 30 <= int(config.VALI_DAYS_FLAG) < 60:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -7
        elif 60 <= int(config.VALI_DAYS_FLAG) < 72:
            train_reward_scaling1 = 2 ** -8
            eval_reward_scaling1 = 2 ** -7
        elif 72 <= int(config.VALI_DAYS_FLAG) < 90:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -7
        elif 90 <= int(config.VALI_DAYS_FLAG) < 100:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -8
        elif 100 <= int(config.VALI_DAYS_FLAG) < 150:
            train_reward_scaling1 = 2 ** -6
            eval_reward_scaling1 = 2 ** -6
        elif 150 <= int(config.VALI_DAYS_FLAG) < 200:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -7
        elif 200 <= int(config.VALI_DAYS_FLAG) < 300:
            train_reward_scaling1 = 2 ** -8
            eval_reward_scaling1 = 2 ** -7
        elif 300 <= int(config.VALI_DAYS_FLAG) < 500:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -7
        elif 500 <= int(config.VALI_DAYS_FLAG) < 518:
            train_reward_scaling1 = 2 ** -8
            eval_reward_scaling1 = 2 ** -8
        elif 518 <= int(config.VALI_DAYS_FLAG) < 1000:
            train_reward_scaling1 = 2 ** -8
            eval_reward_scaling1 = 2 ** -8
        elif 1000 <= int(config.VALI_DAYS_FLAG) < 1200:
            train_reward_scaling1 = 2 ** -8
            eval_reward_scaling1 = 2 ** -8
        elif 1200 <= int(config.VALI_DAYS_FLAG) < 1268:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -8
        elif 1268 <= int(config.VALI_DAYS_FLAG):
            train_reward_scaling1 = 2 ** -9
            eval_reward_scaling1 = 2 ** -9
        else:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -7
            pass
        pass
    elif config.AGENT_NAME == 'AgentDDPG':
        agent_class1 = AgentDDPG()
        if_on_policy1 = False
        break_step1 = 10000

        if 20 <= int(config.VALI_DAYS_FLAG) < 30:
            train_reward_scaling1 = 2 ** -10
            eval_reward_scaling1 = 2 ** -5
            pass
        elif 30 <= int(config.VALI_DAYS_FLAG) < 40:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -6
            pass
        elif 40 <= int(config.VALI_DAYS_FLAG) < 50:
            train_reward_scaling1 = 2 ** -6
            eval_reward_scaling1 = 2 ** -5
            pass
        elif 50 <= int(config.VALI_DAYS_FLAG) < 60:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -5
            pass
        elif 60 <= int(config.VALI_DAYS_FLAG) < 72:
            train_reward_scaling1 = 2 ** -6
            eval_reward_scaling1 = 2 ** -5
            pass
        elif 72 <= int(config.VALI_DAYS_FLAG) < 90:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -7
            pass
        elif 90 <= int(config.VALI_DAYS_FLAG) < 100:
            train_reward_scaling1 = 2 ** -10
            eval_reward_scaling1 = 2 ** -8
            pass
        elif 100 <= int(config.VALI_DAYS_FLAG) < 500:
            train_reward_scaling1 = 2 ** -9
            eval_reward_scaling1 = 2 ** -8
            pass
        elif 500 <= int(config.VALI_DAYS_FLAG):
            train_reward_scaling1 = 2 ** -9
            eval_reward_scaling1 = 2 ** -9
            pass
        else:
            train_reward_scaling1 = 2 ** -7
            eval_reward_scaling1 = 2 ** -6
            pass
        pass
    elif config.AGENT_NAME == 'AgentModSAC':
        agent_class1 = AgentModSAC()
        if_on_policy1 = False
        break_step1 = int(1e6)

        if 20 <= int(config.VALI_DAYS_FLAG) < 30:
            train_reward_scaling1 = 2 ** -12
            eval_reward_scaling1 = 2 ** -9
            pass
        elif 1000 <= int(config.VALI_DAYS_FLAG):
            train_reward_scaling1 = 2 ** -11
            eval_reward_scaling1 = 2 ** -9
            pass
        else:
            train_reward_scaling1 = 2 ** -11
            eval_reward_scaling1 = 2 ** -8
            pass

    elif config.AGENT_NAME == 'AgentDuelingDQN':
        agent_class1 = AgentDuelingDQN()
        if_on_policy1 = False
        break_step1 = 10000
        train_reward_scaling1 = 2 ** -8
        eval_reward_scaling1 = 2 ** -7
        pass
    elif config.AGENT_NAME == 'AgentSharedSAC':
        agent_class1 = AgentSharedSAC()
        if_on_policy1 = False
        break_step1 = 10000
        train_reward_scaling1 = 2 ** -8
        eval_reward_scaling1 = 2 ** -7
        pass
    elif config.AGENT_NAME == 'AgentDoubleDQN':
        agent_class1 = AgentDoubleDQN()
        if_on_policy1 = False
        break_step1 = 10000
        train_reward_scaling1 = 2 ** -8
        eval_reward_scaling1 = 2 ** -7
        pass
    pass

    return agent_class1, if_on_policy1, break_step1, train_reward_scaling1, eval_reward_scaling1


if __name__ == '__main__':

    predict_work_days = 100

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

    config.START_DATE = "2003-05-01"
    config.START_EVAL_DATE = ""

    # 整体结束日期，今天的日期，减去90工作日
    config.END_DATE = str(get_next_work_day(get_datetime_from_date_str(get_today_date()), -predict_work_days))

    # 更新股票数据
    StockData.update_stock_data(tic_code=config.SINGLE_A_STOCK_CODE[0])

    # 好用 AgentPPO(), # AgentSAC(), AgentTD3(), AgentDDPG(), AgentModSAC(),
    # AgentDoubleDQN 单进程好用?
    # 不好用 AgentDuelingDQN(), AgentDoubleDQN(), AgentSharedSAC()

    # 选择agent
    for agent_item in ['AgentTD3', 'AgentDDPG', 'AgentSAC', 'AgentModSAC', 'AgentPPO', ]:

        config.AGENT_NAME = agent_item

        # 预测的截止日期
        end_vali_date = get_datetime_from_date_str(config.END_DATE)

        # 获取7个日期list
        list_begin_vali_date = get_begin_vali_date_list(end_vali_date)

        # 过滤日期，专项预测
        # list_begin_vali_date = filter_date(list_begin_vali_date)
        # print('filter_date', list_begin_vali_date)

        # 倒序，由小到大
        # list_begin_vali_date.reverse()

        # 循环 list_begin_vali_date
        for work_days, begin_date in list_begin_vali_date:

            # 开始的时间
            time_begin = datetime.now()

            # work_days, begin_date = begin_vali_item

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

            # 获得Agent参数
            agent_class, if_on_policy, break_step, train_reward_scaling, eval_reward_scaling = get_agent_args()

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

            args.break_step = break_step
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
    pass

# TODO
# ----------------------------------------
#
# train_reward 表
#
# ----------------------------------------
#
# # AgentTD3
#
# agent_name					AgentTD3
#
# if_on_policy1				False
#
# break_step1					50000
#
# train_reward_scaling1		2 ** (-7)
#
# eval_reward_scaling1		2 ** (-7)
#
# 训练的次数					5
#
# 训练的时间点					2021-07-10 01：21：01
#
# ----------------------------------------
#
#
# 训练之前，根据 Agent 名称 和 VALI_DAYS_FLAG， 读取 train_reward_scaling1 和 eval_reward_scaling1。
#
#
# 训练临时表， temp_train_record ，训练之前，清空。
#
# 训练 的时候，如果 train reward 或 predict reward 大于 256，则 记录此值历史记录，以流水帐的形式记录到 temp_train_record 表。
#
# 训练结束后，判断 temp_train_record 表，是否有记录，如果有，则 整除 256，将此值，更新到 train_reward 表的 train_reward_scaling1 和 eval_reward_scaling1，减去相应的值 例如 (-7 -2)
#
# 循环
#
# ----------------------------------------
#
# 查找有无此2表，创建此2表：train_reward、temp_train_record 。
#
#
# 训练之前，从 train_reward 表 中找到 训练的次数 最小 的记录。
#
# 获取 train_reward_scaling1 和 eval_reward_scaling1 等信息。
#
# 清空 temp_train_record 表，
#
# 训练 的时候，如果 train reward 或 predict reward 大于 256，则 记录此值到 temp_train_record 表。
#
# 训练结束后，拷贝替换 pth 文件。
#
# 训练结束后，train_reward 表 中的 训练的次数 +1，训练的时间点 更新。
#
# 训练结束后，判断 temp_train_record 表，是否有记录，如果有，则 整除 256，将此值，更新到 train_reward 表的 train_reward_scaling1 和 eval_reward_scaling1，减去相应的值 例如 (-7 -2)
#
# 循环
#
#
# ----------------------------------------
