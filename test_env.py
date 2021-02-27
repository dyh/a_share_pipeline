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

    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    # 是否需要下载数据
    is_need_download_data = False
    csv_file_path = "./" + config.DATA_SAVE_DIR + '/' + 'tsla.csv'

    """
    训练agent
    """

    if is_need_download_data:
        print("==============Start Fetching Data===========")
        df = YahooDownloader(
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            # ticker_list=config.DOW_30_TICKER,
            ticker_list=config.SINGLE_TICKER,
        ).fetch_data()

        csv_file_path = "./" + config.DATA_SAVE_DIR + '/' + 'tsla.csv'
        df.to_csv(csv_file_path)

    else:
        # df = pd.read_csv(csv_file_path)
        df = pd.read_csv(csv_file_path).iloc[:, 1:]
        df = df.fillna(0)
        pass

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

    information_cols = ['daily_variance', 'change', 'log_volume', 'close', 'day',
                        'macd', 'rsi_30', 'cci_30', 'dx_30']

    e_train_gym = StockTradingEnvCashpenaltyAShare(df=train, initial_amount=10000, hmax=1000,
                                                   turbulence_threshold=None,
                                                   currency='$',
                                                   buy_cost_pct=3e-3,
                                                   sell_cost_pct=3e-3,
                                                   cash_penalty_proportion=0.2,
                                                   cache_indicator_data=True,
                                                   daily_information_cols=information_cols,
                                                   print_verbosity=500,
                                                   random_start=True)

    e_trade_gym = StockTradingEnvCashpenaltyAShare(df=trade, initial_amount=10000, hmax=1000,
                                                   turbulence_threshold=None,
                                                   currency='$',
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

    model_a2c = agent.get_model("a2c")
    tb_log_name = "a2c"
    total_timesteps = 80000

    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name=tb_log_name, total_timesteps=total_timesteps
    )

    # 保存weights
    weights_file_path = f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{now}"
    trained_a2c.save(weights_file_path)
    print('weights file saved as:', weights_file_path)

    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_a2c, environment=e_trade_gym
    )

    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print('done!')

    # print("==============Get Backtest Results===========")
    # # perf_stats_all = BackTestStats(df_account_value)
    # perf_stats_all = backtest_stats(df_account_value)
    # perf_stats_all = pd.DataFrame(perf_stats_all)
    # perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

    pass
