from stock_data import StockData
# from train_single import get_agent_args
from train_helper import query_model_hyper_parameters_sqlite, query_begin_vali_date_list_by_agent_name
from utils.psqldb import Psqldb
from agent import *
from utils.date_time import *
from env_single import StockTradingEnvSingle, FeatureEngineer
from run_single import *
from datetime import datetime


def calc_max_return(price_ary, initial_capital_temp):
    max_return_temp = 0
    min_value = 0

    assert price_ary.shape[0] > 1

    count_price = price_ary.shape[0]

    for index_left in range(0, count_price - 1):

        for index_right in range(index_left + 1, count_price):

            assert price_ary[index_left][0] > 0

            assert price_ary[index_right][0] > 0

            temp_value = price_ary[index_right][0] - price_ary[index_left][0]

            if temp_value > max_return_temp:
                max_return_temp = temp_value
                # max_value = price_ary[index1][0]
                min_value = price_ary[index_right][0]
            pass
        pass

    if min_value == 0:
        ret = 0
    else:
        ret = (initial_capital_temp / min_value * max_return_temp + initial_capital_temp) / initial_capital_temp

    return ret


if __name__ == '__main__':
    # 预测，并保存结果到 postgresql 数据库
    # 开始预测的时间
    time_begin = datetime.now()

    config.OUTPUT_DATE = '2021-08-03'

    initial_capital = 150000

    max_stock = 3000

    # for tic_item in ['sh.600036', 'sh.600667']:
    # 循环
    for tic_item in ['sh.600036', ]:

        # 要预测的那一天
        config.SINGLE_A_STOCK_CODE = [tic_item, ]

        # psql对象
        psql_object = Psqldb(database=config.PSQL_DATABASE, user=config.PSQL_USER,
                             password=config.PSQL_PASSWORD, host=config.PSQL_HOST, port=config.PSQL_PORT)

        # 好用 AgentPPO(), # AgentSAC(), AgentTD3(), AgentDDPG(), AgentModSAC(),
        # AgentDoubleDQN 单进程好用?
        # 不好用 AgentDuelingDQN(), AgentDoubleDQN(), AgentSharedSAC()
        for agent_item in ['AgentSAC', 'AgentPPO', 'AgentDDPG', 'AgentTD3', 'AgentModSAC', ]:

            config.AGENT_NAME = agent_item
            # config.CWD = f'./{config.AGENT_NAME}/single/{config.SINGLE_A_STOCK_CODE[0]}/StockTradingEnv-v1'

            break_step = int(3e6)

            if_on_policy = False
            # if_use_gae = False

            # 预测的开始日期和结束日期，都固定

            # 日期列表
            # 4月16日向前，20,30,40,50,60,72,90周期
            # end_vali_date = get_datetime_from_date_str('2021-04-16')
            config.IF_ACTUAL_PREDICT = True

            config.START_DATE = "2003-05-01"

            # 前29后1
            config.PREDICT_PERIOD = '60'

            # 固定日期
            config.START_EVAL_DATE = str(get_next_work_day(get_datetime_from_date_str(config.OUTPUT_DATE), -10))
            # config.START_EVAL_DATE = "2021-05-22"

            # OUTPUT_DATE 向右3工作日
            config.END_DATE = str(get_next_work_day(get_datetime_from_date_str(config.OUTPUT_DATE), +1))

            # 创建预测结果表
            StockData.create_predict_result_table_psql(list_tic=config.SINGLE_A_STOCK_CODE)

            # 更新股票数据
            StockData.update_stock_data_to_sqlite(list_stock_code=config.SINGLE_A_STOCK_CODE)

            # 预测的截止日期
            end_vali_date = get_datetime_from_date_str(config.END_DATE)

            # 获取 N 个日期list
            list_begin_vali_date = query_begin_vali_date_list_by_agent_name(agent_item, end_vali_date)

            # 循环 vali_date_list 训练7次
            for vali_days_count, begin_vali_date in list_begin_vali_date:

                # config.START_EVAL_DATE = str(begin_vali_date)

                # 更新工作日标记，用于 run_single.py 加载训练过的 weights 文件
                config.VALI_DAYS_FLAG = str(vali_days_count)

                # config.PREDICT_PERIOD = str(vali_days_count)

                # weights 文件目录
                # model_folder_path = f'./{config.AGENT_NAME}/single/{config.SINGLE_A_STOCK_CODE[0]}' \
                #                     f'/single_{config.VALI_DAYS_FLAG}'
                model_folder_path = f'./{config.WEIGHTS_PATH}/single/{config.AGENT_NAME}/{config.SINGLE_A_STOCK_CODE[0]}' \
                                    f'/single_{config.VALI_DAYS_FLAG}'

                # 如果存在目录则预测
                if os.path.exists(model_folder_path):

                    print('\r\n')
                    print('#' * 40)
                    print('config.AGENT_NAME', config.AGENT_NAME)
                    print('# 预测周期', config.START_EVAL_DATE, '-', config.END_DATE)
                    print('# 模型的 work_days', vali_days_count)
                    print('# model_folder_path', model_folder_path)
                    print('# initial_capital', initial_capital)
                    print('# max_stock', max_stock)

                    initial_stocks = np.zeros(len(config.SINGLE_A_STOCK_CODE), dtype=np.float32)
                    initial_stocks[0] = 100.0

                    # 获取超参
                    model_name = agent_item + '_' + str(vali_days_count)

                    hyper_parameters_id, hyper_parameters_model_name, if_on_policy, break_step, train_reward_scale, \
                    eval_reward_scale, training_times, time_point \
                        = query_model_hyper_parameters_sqlite(model_name=model_name)

                    if if_on_policy == 1:
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

                    args = Arguments(if_on_policy=if_on_policy)
                    args.agent = agent_class

                    args.gpu_id = 0
                    # args.agent.if_use_gae = if_use_gae
                    args.agent.lambda_entropy = 0.04

                    tech_indicator_list = [
                        'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                        'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST

                    gamma = 0.99

                    buy_cost_pct = 0.003
                    sell_cost_pct = 0.003
                    start_date = config.START_DATE
                    start_eval_date = config.START_EVAL_DATE
                    end_eval_date = config.END_DATE

                    args.env = StockTradingEnvSingle(cwd='', gamma=gamma, max_stock=max_stock,
                                                     initial_capital=initial_capital,
                                                     buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct,
                                                     start_date=start_date,
                                                     end_date=start_eval_date, env_eval_date=end_eval_date,
                                                     ticker_list=config.SINGLE_A_STOCK_CODE,
                                                     tech_indicator_list=tech_indicator_list,
                                                     initial_stocks=initial_stocks,
                                                     if_eval=True)

                    args.env_eval = StockTradingEnvSingle(cwd='', gamma=gamma, max_stock=max_stock,
                                                          initial_capital=initial_capital,
                                                          buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct,
                                                          start_date=start_date,
                                                          end_date=start_eval_date, env_eval_date=end_eval_date,
                                                          ticker_list=config.SINGLE_A_STOCK_CODE,
                                                          tech_indicator_list=tech_indicator_list,
                                                          initial_stocks=initial_stocks,
                                                          if_eval=True)

                    args.env.target_return = 100
                    args.env_eval.target_return = 100

                    # 奖励 比例
                    args.env.reward_scaling = train_reward_scaling
                    args.env_eval.reward_scaling = eval_reward_scaling

                    print('train/eval reward scaling:', args.env.reward_scaling, args.env_eval.reward_scaling)

                    # Hyperparameters
                    args.gamma = gamma
                    # ----
                    args.break_step = break_step
                    # ----

                    args.net_dim = 2 ** 9
                    args.max_step = args.env.max_step

                    # ----
                    # args.max_memo = args.max_step * 4
                    args.max_memo = (args.max_step - 1) * 8
                    # ----

                    # ----
                    args.batch_size = 2 ** 12
                    # args.batch_size = 2305
                    # ----

                    # ----
                    # args.repeat_times = 2 ** 3
                    args.repeat_times = 2 ** 4
                    # ----

                    args.eval_gap = 2 ** 4
                    args.eval_times1 = 2 ** 3
                    args.eval_times2 = 2 ** 5

                    # ----
                    args.if_allow_break = False
                    # args.if_allow_break = True
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

                            # 获取要预测的日期，保存到数据库中
                            for item in env.list_buy_or_sell_output:
                                tic, date, action, hold, day, episode_return = item
                                if str(date) == config.OUTPUT_DATE:
                                    # 简单计算一次,低买高卖的最大回报
                                    max_return = calc_max_return(env.price_ary, env.initial_capital)

                                    # 找到要预测的那一天，存储到psql
                                    StockData.update_predict_result_to_psql(psql=psql_object, agent=config.AGENT_NAME,
                                                                            vali_period_value=config.VALI_DAYS_FLAG,
                                                                            pred_period_name=config.PREDICT_PERIOD,
                                                                            tic=tic, date=date, action=action,
                                                                            hold=hold,
                                                                            day=day, episode_return=episode_return,
                                                                            max_return=max_return,
                                                                            trade_detail=env.output_text_trade_detail)

                                    break

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
