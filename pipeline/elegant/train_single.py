import sys

if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../ElegantRL_master')

import shutil

from pipeline.sqlite import SQLite
from pipeline.stock_data import StockData
from pipeline.utils.datetime import get_datetime_from_date_str, time_point, get_begin_vali_date_list, get_next_day
from pipeline.elegant.run_single import *
from pipeline.elegant.agent_single import *
from pipeline.elegant.env_train_single import StockTradingEnv
from pipeline.elegant.env_predict_single import StockTradingEnvPredict, FeatureEngineer

if __name__ == '__main__':
    # AgentPPO(), # AgentSAC(), AgentTD3(), AgentDDPG(), AgentDuelingDQN(), AgentModSAC(), AgentSharedSAC
    # 选择agent
    config.AGENT_NAME = 'AgentPPO'
    config.CWD = f'./{config.AGENT_NAME}/StockTradingEnv-v1'

    if_on_policy = True
    if_use_gae = True

    # 开始训练的日期，在程序启动之后，不要改变。
    config.SINGLE_A_STOCK_CODE = ['sh.600036', ]

    config.IF_SHOW_PREDICT_INFO = False

    config.START_DATE = "2002-05-01"
    config.START_EVAL_DATE = "2021-03-12"
    config.END_DATE = "2021-04-15"

    # 4月16日向前，20,30,40,50,60,72,90周期

    # 预测的截止日期
    end_vali_date = get_datetime_from_date_str(config.END_DATE)

    # 获取7个日期list
    list_begin_vali_date = get_begin_vali_date_list(end_vali_date)

    # 倒序
    # list_begin_vali_date.reverse()

    # 只训练20周期的
    # begin_vali_date = get_next_day(end_vali_date, next_flag=-28)
    # list_begin_vali_date = [(20, begin_vali_date), ]

    # 初始现金
    initial_capital = 20000

    # 单次 购买/卖出 最大股数
    max_stock = 1000

    # 下载、更新 股票数据
    # StockData.update_batch_stock_sqlite(list_stock_code=config.SINGLE_A_STOCK_CODE, dbname=config.STOCK_DB_PATH)

    # # do fe
    # # 缓存 raw 数据 为 df 。
    # raw_df = StockData.load_stock_raw_data_from_sqlite(list_batch_code=config.SINGLE_A_STOCK_CODE,
    #                                                    date_begin=config.START_DATE, date_end=config.END_DATE,
    #                                                    db_path=config.STOCK_DB_PATH)
    #
    # # raw_df -> fe
    # fe_origin_table_name = "fe_origin"
    #
    # # 创建fe表
    # StockData.create_fe_table(db_path=config.STOCK_DB_PATH, table_name=fe_origin_table_name)
    #
    # fe = FeatureEngineer(use_turbulence=False,
    #                      user_defined_feature=False,
    #                      use_technical_indicator=True,
    #                      tech_indicator_list=config.TECHNICAL_INDICATORS_LIST, )
    #
    # fe_df = fe.preprocess_data(raw_df)
    #
    # # 将 fe_df 存入数据库
    # # 先清空，再 insert
    # StockData.clear_and_insert_fe_to_db(fe_df, fe_origin_table_name=fe_origin_table_name)

    # 循环 list_begin_vali_date
    for begin_vali_item in list_begin_vali_date:

        work_days, begin_date = begin_vali_item

        # 更新工作日标记，用于 run_single.py 加载训练过的 weights 文件
        config.WORK_DAY_FLAG = str(work_days)

        model_folder_path = f'./{config.AGENT_NAME}/single_{config.WORK_DAY_FLAG}'

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

        # Agent
        # args = Arguments(if_on_policy=True)
        args = Arguments(if_on_policy=if_on_policy)

        if config.AGENT_NAME == 'AgentPPO':
            args.agent = AgentPPO()
            pass
        elif config.AGENT_NAME == 'AgentSAC':
            args.agent = AgentSAC()
            pass
        elif config.AGENT_NAME == 'AgentTD3':
            args.agent = AgentTD3()
            pass
        elif config.AGENT_NAME == 'AgentDDPG':
            args.agent = AgentDDPG()
            pass
        elif config.AGENT_NAME == 'AgentModSAC':
            args.agent = AgentModSAC()
            pass
        elif config.AGENT_NAME == 'AgentDuelingDQN':
            args.agent = AgentDuelingDQN()
            pass
        elif config.AGENT_NAME == 'AgentSharedSAC':
            args.agent = AgentSharedSAC()
            pass

        # args.agent.if_use_gae = True
        args.agent.if_use_gae = if_use_gae
        args.agent.lambda_entropy = 0.04

        tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
            'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST

        gamma = 0.99
        # max_stock = 1e2
        # initial_capital = 100000
        # initial_stocks = np.zeros(len(config.SINGLE_A_STOCK_CODE), dtype=np.float32)
        initial_stocks_train = np.zeros(len(config.SINGLE_A_STOCK_CODE), dtype=np.float32)
        initial_stocks_vali = np.zeros(len(config.SINGLE_A_STOCK_CODE), dtype=np.float32)

        # 默认持有100股
        initial_stocks_train[0] = 100.0
        initial_stocks_vali[0] = 100.0

        print('# initial_stocks_train', initial_stocks_train)
        print('# initial_stocks_vali', initial_stocks_vali)

        buy_cost_pct = 0.0003
        sell_cost_pct = 0.0003
        start_date = config.START_DATE
        start_eval_date = config.START_EVAL_DATE
        end_eval_date = config.END_DATE

        # train
        args.env = StockTradingEnv(cwd='./datasets', gamma=gamma, max_stock=max_stock,
                                   initial_capital=initial_capital,
                                   buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct, start_date=start_date,
                                   end_date=start_eval_date, env_eval_date=end_eval_date, ticker_list=config.SINGLE_A_STOCK_CODE,
                                   tech_indicator_list=tech_indicator_list, initial_stocks=initial_stocks_train,
                                   if_eval=False)

        # eval
        args.env_eval = StockTradingEnvPredict(cwd='./datasets', gamma=gamma, max_stock=max_stock,
                                               initial_capital=initial_capital,
                                               buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct,
                                               start_date=start_date,
                                               end_date=start_eval_date, env_eval_date=end_eval_date,
                                               ticker_list=config.SINGLE_A_STOCK_CODE,
                                               tech_indicator_list=tech_indicator_list,
                                               initial_stocks=initial_stocks_vali,
                                               if_eval=True)

        args.env.target_reward = 3
        args.env_eval.target_reward = 3

        # Hyperparameters
        args.gamma = gamma
        # ----
        # args.break_step = int(5e6)
        # args.break_step = int(2e6)
        args.break_step = int(3e6)
        # ----

        args.net_dim = 2 ** 9
        args.max_step = args.env.max_step

        # ----
        # args.max_memo = args.max_step * 4
        args.max_memo = (args.max_step - 1) * 8
        # ----

        # ----
        # args.batch_size = 2 ** 10
        args.batch_size = 2 ** 11
        # ----

        # ----
        # args.repeat_times = 2 ** 3
        args.repeat_times = 2 ** 4
        # ----

        args.eval_gap = 2 ** 4
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        # ----
        # args.if_allow_break = False
        args.if_allow_break = True
        # ----

        args.rollout_num = 2  # the number of rollout workers (larger is not always faster)

        # train_and_evaluate(args)
        train_and_evaluate_mp(args)  # the training process will terminate once it reaches the target reward.

        # 保存训练后的模型
        # cwd: ./AgentPPO/StockTradingEnv-v1_0
        model_file_path = f'./{config.AGENT_NAME}/single_{config.WORK_DAY_FLAG}/actor.pth'
        shutil.copyfile(f'{config.CWD}/actor.pth', model_file_path)

        # 保存训练曲线图
        # plot_learning_curve.jpg
        timepoint_temp = time_point()
        plot_learning_curve_file_path = f'./{config.AGENT_NAME}/single_{config.WORK_DAY_FLAG}/plot_{timepoint_temp}.jpg'
        shutil.copyfile(f'{config.CWD}/plot_learning_curve.jpg', plot_learning_curve_file_path)

        # sleep 60 秒
        print('sleep 60 秒')
        time.sleep(60)
        pass
    pass
