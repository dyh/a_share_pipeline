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

    # 股票代码
    stock_code = 'sh.600036'
    # stock_code = 'sz.300513'

    test_csv_file_path = "./" + config.DATA_SAVE_DIR + '/' + stock_code + '.csv'

    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    # （暂不做）持有股票数量

    # 现金金额
    initial_amount = 100000

    # 单支股票最大交易额
    hmax = 10000

    # 买入 手续费
    buy_cost_pct = 3e-3
    # 卖出 手续费
    sell_cost_pct = 3e-3

    # 设置日期，设置open
    end_date = config.END_DATE
    open_price = 52.80

    # 形成待预测的df信息
    df = pd.read_csv(test_csv_file_path)

    # 分割，保留待预测的df
    df = data_split(df, config.START_TRADE_DATE, config.END_DATE)

    # 删除停牌日期的行，即 tradestatus=0 的数据
    df = df[df['tradestatus'] == 1]

    # 删除 tradestatus 列、adjustflag 列
    df.drop(['tradestatus', 'adjustflag', 'isST'], axis=1, inplace=True)

    # 删除未来数据，把df分为2个表，date+open是A表，其余的是B表
    df_left = df.drop(df.columns[2:], axis=1)

    df_right = df.drop(['date', 'open'], axis=1)

    # 删除A表第一行
    df_left.drop(df_left.index[0], inplace=True)
    df_left.reset_index(drop=True, inplace=True)

    # 删除B表最后一行
    # df_right.drop(df_right.index[-1:], inplace=True)
    # df_right.reset_index(drop=True, inplace=True)

    # 将A表和B表重新拼接，即 剔除了未来数据
    df = pd.concat([df_left, df_right], axis=1)

    # 为今天，最后一行赋新值
    #     end_date = config.END_DATE
    #     open_price = 53.60
    df.loc[df.index[-1:], 'date'] = end_date
    df.loc[df.index[-1:], 'open'] = open_price

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

    trade = processed

    # 传入env

    information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'macd', 'rsi_30', 'cci_30', 'dx_30',
                        'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']

    e_trade_gym = StockTradingEnvCashpenaltyAShare(df=trade, initial_amount=initial_amount, hmax=hmax,
                                                   turbulence_threshold=None,
                                                   currency='CNY',
                                                   buy_cost_pct=buy_cost_pct,
                                                   sell_cost_pct=sell_cost_pct,
                                                   cash_penalty_proportion=0.2,
                                                   cache_indicator_data=True,
                                                   daily_information_cols=information_cols,
                                                   print_verbosity=500,
                                                   random_start=False)

    # 加载训练好的weights文件

    # e_trade_gym = StockTradingEnvCashpenalty(df=trade, turbulence_threshold=250, **env_kwargs)
    # env_train, _ = e_train_gym.get_sb_env()
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=e_trade_gym)

    print("==============Model Prediction===========")
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
    # total_timesteps = 80000
    tb_log_name = "a2c"
    model_object = agent.get_model(tb_log_name)
    # -------------------------------------

    # -----------------sac-----------------
    # total_timesteps = 80000
    # tb_log_name = "sac"
    # model_object = agent.get_model(tb_log_name)
    # -------------------------------------

    # trained_model = agent.train_model(model=model_object, tb_log_name=tb_log_name, total_timesteps=total_timesteps)

    trained_model = model_object

    # 保存weights
    weights_file_path = f"{config.TRAINED_MODEL_DIR}/A2C_80k_20210228-16h27.zip"
    # trained_model.save(weights_file_path)

    trained_model.load(weights_file_path)

    print('weights file loaded from:', weights_file_path)

    # 预测

    # 转换预测结果


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
