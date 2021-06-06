from stock_data import StockData
from utils.psqldb import Psqldb
from agent_single import *
from utils.date_time import *
from env_predict_single import StockTradingEnvPredict, FeatureEngineer
from run_single import *
from datetime import datetime

if __name__ == '__main__':

    # 开始预测的时间
    time_begin = datetime.now()

    initial_capital = 100000

    max_stock = 3000

    # for tic_item in ['sh.600036', 'sh.600667']:
    # 循环
    for tic_item in ['sh.600036', ]:

        # 要预测的那一天
        config.SINGLE_A_STOCK_CODE = [tic_item, ]

        # psql对象
        psql_object = Psqldb(database=config.PSQL_DATABASE, user=config.PSQL_USER,
                             password=config.PSQL_PASSWORD, host=config.PSQL_HOST, port=config.PSQL_PORT)

        config.OUTPUT_DATE = '2021-06-07'

        # 前10后10，前10后x，前x后10
        config.PREDICT_PERIOD = '20'

        # 好用 AgentPPO(), # AgentSAC(), AgentTD3(), AgentDDPG(), AgentModSAC(),
        # AgentDoubleDQN 单进程好用?
        # 不好用 AgentDuelingDQN(), AgentDoubleDQN(), AgentSharedSAC()
        # for agent_item in ['AgentModSAC', ]:
        for agent_item in ['AgentPPO', 'AgentSAC', 'AgentTD3', 'AgentDDPG', 'AgentModSAC']:

            config.AGENT_NAME = agent_item
            # config.CWD = f'./{config.AGENT_NAME}/single/{config.SINGLE_A_STOCK_CODE[0]}/StockTradingEnv-v1'

            break_step = int(1e5)

            if_on_policy = True
            if_use_gae = True

            # 预测的开始日期和结束日期，都固定

            # 日期列表
            # 4月16日向前，20,30,40,50,60,72,90周期
            # end_vali_date = get_datetime_from_date_str('2021-04-16')
            config.IF_SHOW_PREDICT_INFO = True

            config.START_DATE = "2002-05-01"

            # 向左10工作日
            config.START_EVAL_DATE = str(get_next_work_day(get_datetime_from_date_str(config.OUTPUT_DATE), -10))
            # 向右10工作日
            config.END_DATE = str(get_next_work_day(get_datetime_from_date_str(config.OUTPUT_DATE), +9))

            # 创建预测结果表
            StockData.create_predict_result_table_psql(tic=config.SINGLE_A_STOCK_CODE[0])

            # 更新股票数据
            StockData.update_stock_data(tic_code=config.SINGLE_A_STOCK_CODE[0])

            # 预测的截止日期
            begin_vali_date = get_datetime_from_date_str(config.START_EVAL_DATE)

            # 获取7个日期list
            list_end_vali_date = get_end_vali_date_list(begin_vali_date)

            # 循环 vali_date_list 训练7次
            for end_vali_item in list_end_vali_date:
                # torch.cuda.empty_cache()

                vali_days, _ = end_vali_item

                # 更新工作日标记，用于 run_single.py 加载训练过的 weights 文件
                config.VALI_DAYS_FLAG = str(vali_days)

                # weights 文件目录
                model_folder_path = f'./{config.AGENT_NAME}/single/{config.SINGLE_A_STOCK_CODE[0]}' \
                                    f'/single_{config.VALI_DAYS_FLAG}'

                # 如果存在目录则预测
                if os.path.exists(model_folder_path):

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
                    initial_stocks[0] = 1000.0

                    buy_cost_pct = 0.0003
                    sell_cost_pct = 0.0003
                    start_date = config.START_DATE
                    start_eval_date = config.START_EVAL_DATE
                    end_eval_date = config.END_DATE

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
                    model_file_path = f'{model_folder_path}/actor.pth'

                    # 如果model存在，则加载
                    if os.path.exists(model_file_path):
                        agent.save_load_model(model_folder_path, if_save=False)

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
                                tic, date, action, hold, day, episode_return = item
                                if str(date) == config.OUTPUT_DATE:
                                    # 找到要预测的那一天，存储到psql
                                    StockData.update_predict_result_to_psql(psql=psql_object, agent=config.AGENT_NAME,
                                                                            vali_period_value=config.VALI_DAYS_FLAG,
                                                                            pred_period_name=config.PREDICT_PERIOD,
                                                                            tic=tic, date=date, action=action,
                                                                            hold=hold,
                                                                            day=day, episode_return=episode_return)
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
            pass

        psql_object.close()
        pass

    # 结束预测的时间
    time_end = datetime.now()
    duration = (time_end - time_begin).seconds
    print('检测耗时', duration, '秒')
    pass
