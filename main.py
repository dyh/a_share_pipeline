import os
import sys
import datetime
import pandas as pd

from env_stocktrading_a_share import StockTradingEnvAShare
import config_a_share as config

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

    """
    训练agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        # ticker_list=config.DOW_30_TICKER,
        ticker_list=config.SINGLE_TICKER,
    ).fetch_data()
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
            1
            + 2 * stock_dimension
            + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnvAShare(df=train, **env_kwargs)

    e_trade_gym = StockTradingEnvAShare(df=trade, turbulence_threshold=250, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    model_sac = agent.get_model("a2c")
    tb_log_name = "a2c"
    total_timesteps = 80000

    trained_sac = agent.train_model(
        model=model_sac, tb_log_name=tb_log_name, total_timesteps=total_timesteps
    )

    # 保存weights
    weights_file_path = f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{now}"
    trained_sac.save(weights_file_path)
    print('weights file saved as:', weights_file_path)

    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, environment=e_trade_gym
    )

    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest Results===========")
    # perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = backtest_stats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

    pass
