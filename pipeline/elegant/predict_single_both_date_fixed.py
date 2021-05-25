import sys

if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../ElegantRL_master')

from pipeline.utils.psqldb import Psqldb
from pipeline.stock_data import StockData
from pipeline.elegant.agent_single import *
from pipeline.utils.datetime import get_datetime_from_date_str, get_begin_vali_date_list, get_end_vali_date_list
from pipeline.elegant.env_predict_single import StockTradingEnvPredict, FeatureEngineer
from pipeline.elegant.run_single import *


def update_stock_data(date_append_to_raw_df='', tic_code=''):

    # # 要预测的股票
    # config.BATCH_A_STOCK_CODE = ['sh.600036', ]
    #
    # # 默认初始数据日期，因为距离预测开始和结束时间非常远，对预测无影响
    # config.START_DATE = "2002-05-01"
    #
    # # 要预测的日期
    # config.END_DATE = '2021-05-24'

    # 下载、更新 股票数据
    StockData.update_batch_stock_sqlite(list_stock_code=config.SINGLE_A_STOCK_CODE, dbname=config.STOCK_DB_PATH)

    # 缓存 raw 数据 为 df
    raw_df = StockData.load_stock_raw_data_from_sqlite(list_batch_code=config.SINGLE_A_STOCK_CODE,
                                                       date_begin=config.START_DATE, date_end=config.END_DATE,
                                                       db_path=config.STOCK_DB_PATH)

    open1, high1, low1, close1, volume1 = StockData.get_today_stock_data_from_sina_api(
        tic_code=tic_code.replace('.', ''),
        date=date_append_to_raw_df)

    # 查询raw_df里是否有 date 日期的数据，如果没有，则添加临时真实数据
    if raw_df.loc[raw_df["date"] == date_append_to_raw_df].empty is True:
        # 为 raw 添加今日行情数据
        list1 = [(date_append_to_raw_df, open1, high1, low1, close1, volume1, tic_code), ]
        raw_df = StockData.append_rows_to_raw_df(raw_df, list1)
        pass

    # raw_df -> fe
    fe_origin_table_name = "fe_origin"

    # 创建fe表
    StockData.create_fe_table(db_path=config.STOCK_DB_PATH, table_name=fe_origin_table_name)

    fe = FeatureEngineer(use_turbulence=False,
                         user_defined_feature=False,
                         use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST, )

    fe_df = fe.preprocess_data(raw_df)

    # 将 fe_df 存入数据库
    # 增量fe
    # StockData.save_fe_to_db(fe_df, fe_origin_table_name=fe_origin_table_name, dbname=config.STOCK_DB_PATH)
    StockData.clear_and_insert_fe_to_db(fe_df, fe_origin_table_name=fe_origin_table_name)

    pass


if __name__ == '__main__':

    # psql对象
    psql_object = Psqldb(database=config.PSQL_DATABASE, user=config.PSQL_USER,
                         password=config.PSQL_PASSWORD, host=config.PSQL_HOST, port=config.PSQL_PORT)

    config.OUTPUT_DATE = '2021-05-25'

    # 前10后10，前10后x，前x后10
    config.PREDICT_PERIOD = '前10后10'

    # AgentPPO(), # AgentSAC(), AgentTD3(), AgentDDPG(), AgentDuelingDQN(), AgentModSAC(), AgentSharedSAC
    config.AGENT_NAME = 'AgentPPO'
    config.CWD = f'./{config.AGENT_NAME}/StockTradingEnv-v1'
    break_step = int(1e5)

    if_on_policy = True
    if_use_gae = True

    # 预测的开始日期和结束日期，都固定

    # 日期列表
    # 4月16日向前，20,30,40,50,60,72,90周期
    # end_vali_date = get_datetime_from_date_str('2021-04-16')
    config.IF_SHOW_PREDICT_INFO = True

    # 要预测的那一天
    config.SINGLE_A_STOCK_CODE = ['sh.600036', ]

    config.START_DATE = "2002-05-01"
    # config.START_EVAL_DATE = "2021-04-16"
    # config.END_DATE = '2021-05-14'
    config.START_EVAL_DATE = "2021-05-10"
    config.END_DATE = "2021-06-04"

    # 创建预测结果表
    # StockData.create_predict_result_table_psql(tic=config.SINGLE_A_STOCK_CODE[0])

    # 更新股票数据
    update_stock_data(date_append_to_raw_df=config.OUTPUT_DATE, tic_code=config.SINGLE_A_STOCK_CODE[0])

    # 预测的截止日期
    begin_vali_date = get_datetime_from_date_str(config.START_EVAL_DATE)

    # 获取7个日期list
    list_end_vali_date = get_end_vali_date_list(begin_vali_date)

    initial_capital = 20000

    # 循环 vali_date_list 训练7次
    for end_vali_item in list_end_vali_date:
        # torch.cuda.empty_cache()

        # 从100到1k
        max_stock = 1000

        vali_days, _ = end_vali_item

        # 更新工作日标记，用于 run_single.py 加载训练过的 weights 文件
        config.VALI_DAYS_FLAG = str(vali_days)

        # weights 文件目录
        model_folder_path = f'./{config.AGENT_NAME}/single_{config.VALI_DAYS_FLAG}'

        # 如果存在目录则预测
        if os.path.exists(model_folder_path):

            # 停止预测的日期
            # config.END_DATE = str(end_eval_date)

            print('\r\n')
            print('#' * 40)
            print('config.AGENT_NAME', config.AGENT_NAME)
            print('# 预测周期', config.START_EVAL_DATE, '-', config.END_DATE)
            print('# 模型的 work_days', vali_days)
            print('# model_folder_path', model_folder_path)
            print('# initial_capital', initial_capital)
            print('# max_stock', max_stock)

            # Agent
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

            args.gpu_id = 0
            args.agent.if_use_gae = if_use_gae
            args.agent.lambda_entropy = 0.04

            tech_indicator_list = [
                'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST

            gamma = 0.99
            # max_stock = 1e2
            # max_stock = 100
            # initial_capital = 100000
            initial_stocks = np.zeros(len(config.SINGLE_A_STOCK_CODE), dtype=np.float32)
            initial_stocks[0] = 100.0

            buy_cost_pct = 0.0003
            sell_cost_pct = 0.0003
            start_date = config.START_DATE
            start_eval_date = config.START_EVAL_DATE
            end_eval_date = config.END_DATE

            # args.env = StockTradingEnv(cwd='./datasets', gamma=gamma, max_stock=max_stock,
            #                            initial_capital=initial_capital,
            #                            buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct, start_date=start_date,
            #                            end_date=start_eval_date, env_eval_date=end_eval_date, ticker_list=tickers,
            #                            tech_indicator_list=tech_indicator_list, initial_stocks=initial_stocks,
            #                            if_eval=False)

            args.env = StockTradingEnvPredict(cwd='./datasets', gamma=gamma, max_stock=max_stock,
                                              initial_capital=initial_capital,
                                              buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct,
                                              start_date=start_date,
                                              end_date=start_eval_date, env_eval_date=end_eval_date,
                                              ticker_list=config.SINGLE_A_STOCK_CODE,
                                              tech_indicator_list=tech_indicator_list,
                                              initial_stocks=initial_stocks,
                                              if_eval=True)

            args.env_eval = StockTradingEnvPredict(cwd='./datasets', gamma=gamma, max_stock=max_stock,
                                                   initial_capital=initial_capital,
                                                   buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct,
                                                   start_date=start_date,
                                                   end_date=start_eval_date, env_eval_date=end_eval_date,
                                                   ticker_list=config.SINGLE_A_STOCK_CODE,
                                                   tech_indicator_list=tech_indicator_list,
                                                   initial_stocks=initial_stocks,
                                                   if_eval=True)

            args.env.target_reward = 3
            args.env_eval.target_reward = 3

            # Hyperparameters
            args.gamma = gamma
            # ----
            # args.break_step = int(5e6)
            # args.break_step = int(3e6)
            args.break_step = break_step
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

            # ----------------------------
            args.init_before_training()

            '''basic arguments'''
            cwd = args.cwd
            env = args.env
            agent = args.agent
            gpu_id = args.gpu_id  # necessary for Evaluator?

            '''training arguments'''
            net_dim = args.net_dim
            max_memo = args.max_memo
            break_step = args.break_step
            batch_size = args.batch_size
            target_step = args.target_step
            repeat_times = args.repeat_times
            if_break_early = args.if_allow_break
            if_per = args.if_per
            gamma = args.gamma
            reward_scale = args.reward_scale

            '''evaluating arguments'''
            eval_gap = args.eval_gap
            eval_times1 = args.eval_times1
            eval_times2 = args.eval_times2
            if args.env_eval is not None:
                env_eval = args.env_eval
            elif args.env_eval in set(gym.envs.registry.env_specs.keys()):
                env_eval = PreprocessEnv(gym.make(env.env_name))
            else:
                env_eval = deepcopy(env)

            del args  # In order to show these hyper-parameters clearly, I put them above.

            '''init: environment'''
            max_step = env.max_step
            state_dim = env.state_dim
            action_dim = env.action_dim
            if_discrete = env.if_discrete

            '''init: Agent, ReplayBuffer, Evaluator'''
            agent.init(net_dim, state_dim, action_dim, if_per)

            # ----
            # work_days，周期数，用于存储和提取训练好的模型
            model_file_path = f'./{config.AGENT_NAME}/single_{config.VALI_DAYS_FLAG}/actor.pth'
            # 如果model存在，则加载
            if os.path.exists(model_file_path):
                agent.save_load_model(f'./{config.AGENT_NAME}/single_{config.VALI_DAYS_FLAG}', if_save=False)

                # if_on_policy = getattr(agent, 'if_on_policy', False)
                #
                # buffer = ReplayBuffer(max_len=max_memo + max_step, state_dim=state_dim,
                #                       action_dim=1 if if_discrete else action_dim,
                #                       if_on_policy=if_on_policy, if_per=if_per, if_gpu=True)
                #
                # evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2, )

                '''prepare for training'''
                agent.state = env.reset()

                episode_return = 0.0  # sum of rewards in an episode
                episode_step = 1
                max_step = env.max_step
                if_discrete = env.if_discrete

                state = env.reset()

                with torch.no_grad():  # speed up running

                    # for episode_step in range(max_step):
                    while True:
                        s_tensor = torch.as_tensor((state,), device=agent.device)
                        a_tensor = agent.act(s_tensor)
                        if if_discrete:
                            a_tensor = a_tensor.argmax(dim=1)
                        action = a_tensor.detach().cpu().numpy()[
                            0]  # not need detach(), because with torch.no_grad() outside
                        state, reward, done, _ = env.step(action)
                        episode_return += reward
                        if done:
                            break
                        pass
                    pass

                    print('>>>> env.list_output', env.list_buy_or_sell_output)

                    # 插入数据库
                    # tic, date, -sell/+buy, hold, 第x天 = env.list_buy_or_sell_output
                    # agent，vali_days，pred_period = config.AGENT_NAME, config.VALI_DAYS_FLAG, config.PREDICT_PERIOD

                    # 获取要预测的日期，保存到数据库中
                    for item in env.list_buy_or_sell_output:
                        tic, date, action, hold, day = item
                        if str(date) == config.OUTPUT_DATE:
                            # 找到要预测的那一天，存储到psql
                            StockData.push_predict_result_to_psql(psql=psql_object, agent=config.AGENT_NAME,
                                                                  vali_period_value=config.VALI_DAYS_FLAG,
                                                                  pred_period_name=config.PREDICT_PERIOD,
                                                                  tic=tic, date=date, action=action, hold=hold, day=day)
                            pass
                        pass
                    pass

                    # episode_return = getattr(env, 'episode_return', episode_return)

                pass
            else:
                print('未找到模型文件', model_file_path)
            pass
            # ----

        pass

    psql_object.close()

    pass
