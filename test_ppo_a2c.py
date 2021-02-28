import os
import sys
import datetime
import pandas as pd
import numpy as np

import config_a_share as config
from env_stocktrading_cashpenalty_a_share import StockTradingEnvCashpenaltyAShare

sys.path.append('./FinRL_Library_master')

from FinRL_Library_master.finrl.marketdata.yahoodownloader import YahooDownloader
from FinRL_Library_master.finrl.model.models import DRLAgent
from FinRL_Library_master.finrl.preprocessing.data import data_split
from FinRL_Library_master.finrl.preprocessing.preprocessors import FeatureEngineer
from FinRL_Library_master.finrl.trade.backtest import backtest_stats

if __name__ == "__main__":

    # 是否需要下载数据
    is_need_download_data = False

    # stock_code = 'sh.600036'
    stock_code = 'sz.300513'

    test_csv_file_path = "./" + config.DATA_SAVE_DIR + '/' + stock_code + '.csv'

    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    """
    训练agent
    """

    if is_need_download_data:
        print("==============下载数据===========")
    else:
        # df = pd.read_csv(test_csv_file_path).iloc[:, 1:]
        # df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
        # df.fillna(0, inplace=True)
        pass
    pass

    # 开盘价为T日数据，其余皆为T-1日数据，避免引入未来数据

    print("==============使用本地数据===========")
    df = pd.read_csv(test_csv_file_path)

    # 删除停牌日期的行，即 tradestatus=0 的数据
    df = df[df['tradestatus'] == 1]

    # 删除 tradestatus 列、adjustflag 列
    df.drop(['tradestatus', 'adjustflag'], axis=1, inplace=True)

    # 删除未来数据，把df分为2个表，date+open是A表，其余的是B表
    df_left = df.drop(df.columns[2:], axis=1)

    df_right = df.drop(['date', 'open'], axis=1)

    # 删除A表第一行
    df_left.drop(df_left.index[0], inplace=True)
    df_left.reset_index(drop=True, inplace=True)

    # 删除B表最后一行
    df_right.drop(df_right.index[-1:], inplace=True)
    df_right.reset_index(drop=True, inplace=True)

    # 将A表和B表重新拼接，即 剔除了未来数据
    df = pd.concat([df_left, df_right], axis=1)

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    processed['log_volume'] = np.log(processed.volume * processed.close)
    processed['change'] = (processed.close - processed.open) / processed.close
    processed['daily_variance'] = (processed.high - processed.low) / processed.close
    processed.head()

    processed.to_csv(f'{config.DATA_SAVE_DIR}/{stock_code}_temp.csv', index=False)

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    # stock_dimension = len(train.tic.unique())
    # state_space = (
    #         1
    #         + 2 * stock_dimension
    #         + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    # )

    # env_kwargs = {
    #     "hmax": 100,
    #     "initial_amount": 10000,
    #     "buy_cost_pct": 0.001,
    #     "sell_cost_pct": 0.001,
    #     "state_space": state_space,
    #     "stock_dim": stock_dimension,
    #     "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    #     "action_space": stock_dimension,
    #     "reward_scaling": 1e-4
    # }

    # e_train_gym = StockTradingEnvCashpenalty(df=train, **env_kwargs)

    # information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'macd', 'rsi_30', 'cci_30', 'dx_30']

    # fields = "date,open,high,low,close,volume,code,amount," \
    #               "adjustflag,turn,tradestatus,pctChg,peTTM," \
    #               "pbMRQ,psTTM,pcfNcfTTM,isST"

    # information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'macd', 'rsi_30', 'cci_30', 'dx_30']
    # information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'day', 'macd', 'rsi_30', 'cci_30', 'dx_30']

    information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'macd', 'rsi_30', 'cci_30', 'dx_30',
                        'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']

    e_train_gym = StockTradingEnvCashpenaltyAShare(df=train, initial_amount=100000, hmax=1000,
                                                   turbulence_threshold=None,
                                                   currency='CNY',
                                                   buy_cost_pct=3e-3,
                                                   sell_cost_pct=3e-3,
                                                   cash_penalty_proportion=0.2,
                                                   cache_indicator_data=True,
                                                   daily_information_cols=information_cols,
                                                   print_verbosity=500,
                                                   random_start=True)

    e_trade_gym = StockTradingEnvCashpenaltyAShare(df=trade, initial_amount=100000, hmax=1000,
                                                   turbulence_threshold=None,
                                                   currency='CNY',
                                                   buy_cost_pct=3e-3,
                                                   sell_cost_pct=3e-3,
                                                   cash_penalty_proportion=0.2,
                                                   cache_indicator_data=True,
                                                   daily_information_cols=information_cols,
                                                   print_verbosity=500,
                                                   random_start=False)

    # e_trade_gym = StockTradingEnvCashpenalty(df=trade, turbulence_threshold=250, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    # -----------------PPO-----------------
    # total_timesteps = 100000
    # tb_log_name = "ppo"
    #
    # PPO_PARAMS = {
    #     "n_steps": 2048,
    #     "ent_coef": 0.01,
    #     "learning_rate": 0.00025,
    #     "batch_size": 128,
    # }
    #
    # model_object = agent.get_model(tb_log_name, model_kwargs=PPO_PARAMS)
    # -------------------------------------

    # -----------------a2c-----------------
    total_timesteps = 80000
    tb_log_name = "a2c"
    model_object = agent.get_model(tb_log_name)
    # -------------------------------------

    # -----------------sac-----------------
    # total_timesteps = 80000
    # tb_log_name = "sac"
    # model_object = agent.get_model(tb_log_name)
    # -------------------------------------

    trained_model = agent.train_model(model=model_object, tb_log_name=tb_log_name, total_timesteps=total_timesteps)

    # 保存weights
    weights_file_path = f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{now}"
    trained_model.save(weights_file_path)
    print('weights file saved as:', weights_file_path)

    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=e_trade_gym)

    df_account_value.to_csv("./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv")
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print('done!')

    # print("==============Get Backtest Results===========")
    # # perf_stats_all = BackTestStats(df_account_value)
    # perf_stats_all = backtest_stats(df_account_value)
    # perf_stats_all = pd.DataFrame(perf_stats_all)
    # perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

    pass
