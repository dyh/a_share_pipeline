import sys


if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../ElegantRL_master')

import numpy as np

from pipeline.stock_data import StockData

from pipeline.finrl import config
from elegantrl.run import *
from elegantrl.agent import AgentPPO, AgentDDPG
from env_train import StockTradingEnv

if __name__ == '__main__':
    # Agent
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()  # AgentSAC(), AgentTD3(), AgentDDPG()
    args.agent.if_use_gae = True
    args.agent.lambda_entropy = 0.04

    # 沪深300
    config.HS300_CODE_LIST = StockData.get_hs300_code_list()

    tickers = config.HS300_CODE_LIST

    tech_indicator_list = [
        'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
        'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST

    gamma = 0.99
    # max_stock = 1e2
    max_stock = 20000
    initial_capital = 100000
    initial_stocks = np.zeros(len(tickers), dtype=np.float32)
    buy_cost_pct = 0.0003
    sell_cost_pct = 0.0003
    start_date = config.START_DATE
    start_eval_date = config.START_EVAL_DATE
    end_eval_date = config.END_DATE

    args.env = StockTradingEnv(cwd='./datasets', gamma=gamma, max_stock=max_stock, initial_capital=initial_capital,
                               buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct, start_date=start_date,
                               end_date=start_eval_date, env_eval_date=end_eval_date, ticker_list=tickers,
                               tech_indicator_list=tech_indicator_list, initial_stocks=initial_stocks,
                               if_eval=False)

    args.env_eval = StockTradingEnv(cwd='./datasets', gamma=gamma, max_stock=max_stock,
                                    initial_capital=initial_capital,
                                    buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct, start_date=start_date,
                                    end_date=start_eval_date, env_eval_date=end_eval_date, ticker_list=tickers,
                                    tech_indicator_list=tech_indicator_list, initial_stocks=initial_stocks,
                                    if_eval=True)

    args.env.target_reward = 3
    args.env_eval.target_reward = 3

    # Hyperparameters
    args.gamma = gamma
    # ----
    # args.break_step = int(2e5)
    # args.break_step = int(5e6)
    args.break_step = int(2e6)
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

    pass
