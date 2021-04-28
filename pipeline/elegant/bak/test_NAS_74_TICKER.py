import sys


if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../ElegantRL_master')

import numpy as np

from elegantrl.run import *
from elegantrl.agent import AgentPPO, AgentDDPG
from StockTrading import StockTradingEnv, check_stock_trading_env
import yfinance as yf
from stockstats import StockDataFrame as Sdf

import pandas as pd
from pipeline.stock_data import StockData

if __name__ == '__main__':
    # df1 = pd.read_csv('./datasets/StockTradingEnv_raw_data.csv', index_col=0)
    # print(df1)
    #
    # df2 = StockData.fill_zero_value_to_null_date(df=df1, code_list=['AAPL', 'TSLA'],
    #                                              date_column_name='date',
    #                                              code_column_name='tic')
    # df1.close()
    #
    # df2.to_csv('./datasets/StockTradingEnv_raw_data.csv')
    # df2.close()

    # Agent
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()  # AgentSAC(), AgentTD3(), AgentDDPG()
    args.agent.if_use_gae = True
    args.agent.lambda_entropy = 0.04

    # Environment
    # tickers = [
    #     'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN',
    #     'AMZN', 'ASML', 'ATVI', 'BIIB', 'BKNG', 'BMRN', 'CDNS', 'CERN', 'CHKP', 'CMCSA',
    #     'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'FAST',
    #     'FISV', 'GILD', 'HAS', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG',
    #     'JBHT', 'KLAC', 'LRCX', 'MAR', 'MCHP', 'MDLZ', 'MNST', 'MSFT', 'MU', 'MXIM',
    #     'NLOK', 'NTAP', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'QCOM', 'REGN',
    #     'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS', 'TTWO', 'TXN', 'VRSN', 'VRTX', 'WBA',
    #     'WDC', 'WLTW', 'XEL', 'XLNX']  # finrl.config.NAS_74_TICKER

    tickers = ['AAPL', 'TSLA']  # finrl.config.NAS_74_TICKER

    tech_indicator_list = [
        'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
        'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST

    gamma = 0.99
    max_stock = 1e2
    initial_capital = 1e5
    initial_stocks = np.zeros(len(tickers), dtype=np.float32)
    buy_cost_pct = 1e-3
    sell_cost_pct = 1e-3
    start_date = '2010-07-01'
    start_eval_date = '2020-01-01'
    end_eval_date = '2021-04-14'

    args.env = StockTradingEnv('./datasets/', gamma, max_stock, initial_capital, buy_cost_pct,
                               sell_cost_pct, start_date, start_eval_date,
                               end_eval_date, tickers, tech_indicator_list,
                               initial_stocks, if_eval=False)
    args.env_eval = StockTradingEnv('./datasets/', gamma, max_stock, initial_capital, buy_cost_pct,
                                    sell_cost_pct, start_date, start_eval_date,
                                    end_eval_date, tickers, tech_indicator_list,
                                    initial_stocks, if_eval=True)

    args.env.target_reward = 3
    args.env_eval.target_reward = 3

    # Hyperparameters
    args.gamma = gamma
    args.break_step = int(2e5)
    args.net_dim = 2 ** 9
    args.max_step = args.env.max_step
    args.max_memo = args.max_step * 4
    args.batch_size = 2 ** 10
    args.repeat_times = 2 ** 3
    args.eval_gap = 2 ** 4
    args.eval_times1 = 2 ** 3
    args.eval_times2 = 2 ** 5
    args.if_allow_break = False
    args.rollout_num = 2  # the number of rollout workers (larger is not always faster)

    # train_and_evaluate(args)
    train_and_evaluate_mp(args)  # the training process will terminate once it reaches the target reward.

    pass
