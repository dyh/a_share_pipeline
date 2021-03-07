import os
import sys
import time

import pandas as pd
import numpy as np

import config as config
from stock_data import StockData

sys.path.append('./FinRL_Library_master')

from models import DRLAgent
from FinRL_Library_master.finrl.preprocessing.preprocessors import FeatureEngineer

from env import StockTradingAShareEnv as StockTradingEnv

if __name__ == "__main__":

    # 创建目录
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    # 股票代码
    stock_code = 'sh.600036'

    # 训练开始日期
    start_date = "2002-05-01"
    # 停止训练日期 / 开始预测日期
    start_trade_date = "2020-11-16"
    # 停止预测日期
    end_date = '2021-03-04'

    # 今天日期，交易日
    today_date = '2021-03-05'

    # 今日开盘价
    today_open_price = 51.90

    # 现金金额
    initial_amount = 100000

    # 单支股票最大交易量
    hmax = 1000

    # 训练次数
    total_timesteps = 80000

    print("==============下载A股数据==============")

    # 下载A股的日K线数据
    stock_data = StockData("./" + config.DATA_SAVE_DIR, date_start=start_date, date_end=end_date)
    # 获得数据文件路径
    csv_file_path = stock_data.download(stock_code)

    print("==============处理未来数据==============")

    # open今日开盘价为T日数据，其余皆为T-1日数据，避免引入未来数据
    df = pd.read_csv(csv_file_path)

    # 删除未来数据，把df分为2个表，日期date+开盘open是A表，其余的是B表
    df_left = df.drop(df.columns[2:], axis=1)

    df_right = df.drop(['date', 'open'], axis=1)

    # 删除A表第一行
    df_left.drop(df_left.index[0], inplace=True)
    df_left.reset_index(drop=True, inplace=True)

    # # 删除B表最后一行
    # df_right.drop(df_right.index[-1:], inplace=True)
    # df_right.reset_index(drop=True, inplace=True)

    # 将A表和B表重新拼接，剔除了未来数据
    df = pd.concat([df_left, df_right], axis=1)

    # 今天的数据，date、open为空，重新赋值
    df.loc[df.index[-1:], 'date'] = today_date
    df.loc[df.index[-1:], 'open'] = today_open_price

    # 缓存文件，debug用
    df.to_csv(f'{config.DATA_SAVE_DIR}/{stock_code}_concat_df.csv', index=False)

    print("==============加入技术指标==============")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
    )

    df_fe = fe.preprocess_data(df)
    df_fe['log_volume'] = np.log(df_fe.volume * df_fe.close)
    df_fe['change'] = (df_fe.close - df_fe.open) / df_fe.close
    df_fe['daily_variance'] = (df_fe.high - df_fe.low) / df_fe.close

    df_fe.to_csv(f'{config.DATA_SAVE_DIR}/{stock_code}_processed_df.csv', index=False)

    print("==============拆分 训练/预测 数据集==============")
    # Training & Trading data split
    df_train = df_fe[(df_fe.date >= start_date) & (df_fe.date < start_trade_date)]
    df_train = df_train.sort_values(["date", "tic"], ignore_index=True)
    df_train.index = df_train.date.factorize()[0]

    df_predict = df_fe[(df_fe.date >= start_trade_date) & (df_fe.date <= today_date)]
    df_predict = df_predict.sort_values(["date", "tic"], ignore_index=True)
    df_predict.index = df_predict.date.factorize()[0]

    # 训练/预测 时间点
    time_point = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    print("==============开始训练模型==============")

    # weights文件名
    weights_file_path = f"{config.TRAINED_MODEL_DIR}/{time_point}.zip"
    # weights_file_path = f"{config.TRAINED_MODEL_DIR}/20210307_181715.zip"

    # calculate state action space
    stock_dimension = len(df_train.tic.unique())
    state_space = (
            1
            + 2 * stock_dimension
            + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "buy_cost_pct": 0.003,
        "sell_cost_pct": 0.003,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "model_name": 'a2c',
        "mode": 'normal_env',
        "iteration": time_point
    }

    e_train_gym = StockTradingEnv(df=df_train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent_train = DRLAgent(env=env_train)

    # 获取模型定义
    # -----------------PPO-----------------
    # tb_log_name = "ppo"
    # model_object = agent.get_model(tb_log_name, model_kwargs=config.PPO_PARAMS)
    # -------------------------------------

    # -----------------a2c-----------------
    model_name = "a2c"
    model_object = agent_train.get_model(model_name=model_name, model_kwargs=config.A2C_PARAMS)
    # -------------------------------------

    # -----------------sac-----------------
    # tb_log_name = "sac"
    # model_object = agent.get_model(tb_log_name, model_kwargs=config.SAC_PARAMS)
    # -------------------------------------

    # 加载训练好的weights文件，可接续上次的结果继续训练。
    # weights_file_path = f"{config.TRAINED_MODEL_DIR}/a2c.zip"
    # model_object.load(weights_file_path)

    # 训练模型
    model_object = agent_train.train_model(model=model_object, tb_log_name=model_name, total_timesteps=total_timesteps)

    # 保存训练好的weights文件
    model_object.save(weights_file_path)

    print('weights文件保存在:', weights_file_path)
    print("==============模型训练完成==============")

    # 预测
    print("==============开始预测==============")

    # debug
    df_predict.to_csv(f'{config.DATA_SAVE_DIR}/{stock_code}_trade_df.csv', index=False)

    # 开启 湍流阈值
    # e_trade_gym = StockTradingEnv(df=df_trade, turbulence_threshold=250, **env_kwargs)

    # 不使用 湍流阈值
    e_trade_gym = StockTradingEnv(df=df_predict, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    agent_predict = DRLAgent(env=env_trade)

    model_predict = agent_predict.get_model(model_name, model_kwargs=config.A2C_PARAMS)

    # 加载训练好的weights文件
    model_predict.load(weights_file_path)

    df_account_value, df_actions = DRLAgent.DRL_today_prediction(model=model_predict, environment=e_trade_gym)

    account_csv_file_path = "./" + config.RESULTS_DIR + "/df_account_value_" + time_point + ".csv"
    df_account_value.to_csv(account_csv_file_path)

    actions_csv_file_path = "./" + config.RESULTS_DIR + "/df_actions_" + time_point + ".csv"
    df_actions.to_csv(actions_csv_file_path)

    print("account 结果保存在:", account_csv_file_path)
    print("actions 结果保存在:", actions_csv_file_path)

    print("==============预测完成==============")

    pass
