import os
import numpy as np

from pipeline.finrl import config
from pipeline.stock_data import StockData


class StockTradingEnv:
    def __init__(self, beg_i=0, end_i=3220, initial_amount=1e6, initial_stocks=None,
                 max_stock=1e2, buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99,
                 ticker_list=None, tech_id_list=None, beg_date=None, end_date=None, ):
        # load data
        self.close_ary, self.tech_ary = self.get_close_ary_tech_ary(
            ticker_list, tech_id_list, beg_date, end_date, )

        self.close_ary = self.close_ary[beg_i:end_i]
        self.tech_ary = self.tech_ary[beg_i:end_i]

        stock_dim = self.close_ary.shape[1]

        self.max_stock = max_stock
        self.buy_cost_rate = 1 + buy_cost_pct
        self.sell_cost_rate = 1 - sell_cost_pct
        self.initial_amount = initial_amount
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks

        self.max_day = len(self.close_ary)
        self.gamma = gamma

        # reset()
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.episode_return = 0

        self.amount = None
        self.stocks = None

        # environment information
        self.env_name = 'StockTradingEnv-v1'
        self.state_dim = len(self.reset())
        self.action_dim = stock_dim
        self.if_discrete = False
        self.target_return = 20000.0
        self.max_step = len(self.close_ary)

    def reset(self):
        self.day = 0
        self.rewards = list()
        self.amount = self.initial_amount
        self.stocks = self.initial_stocks

        self.total_asset = (self.close_ary[self.day] * self.stocks).sum() + self.amount

        state = np.array((self.amount, *self.stocks,
                          *self.close_ary[self.day],
                          *self.tech_ary[self.day],), dtype=np.float32)
        return state

    def step(self, action):
        self.day += 1
        done = self.day == self.max_day - 1

        action = action * self.max_stock  # actions initially is scaled between 0 to 1
        action = (action.astype(int))  # convert into integer because we can't by fraction of shares

        # sort_actions = np.argsort(action)
        # sell_index = sort_actions[:np.where(action < 0)[0].shape[0]]
        # buy_index = sort_actions[::-1][:np.where(action > 0)[0].shape[0]]
        #
        # for index in sell_index:
        #     action[index] = self._sell_stock(index, action[index]) * (-1)
        # for index in buy_index:
        #     action[index] = self._buy_stock(index, action[index])

        for index in range(self.action_dim):
            stock_action = action[index]
            adj_close_price = self.close_ary[self.day, index]  # `adjcp` denotes adjusted close price?
            if stock_action > 0:  # buy_stock
                delta_stock = min(self.amount // adj_close_price, stock_action)
                self.amount -= adj_close_price * delta_stock * self.buy_cost_rate
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.stocks[index])
                self.amount += adj_close_price * delta_stock * self.sell_cost_rate
                self.stocks[index] -= delta_stock

        state = np.array((self.amount, *self.stocks,
                          *self.close_ary[self.day],
                          *self.tech_ary[self.day],), dtype=np.float32)

        total_asset = (self.close_ary[self.day] * self.stocks).sum() + self.amount
        # reward = (total_asset - self.total_asset) * 2 ** -6
        reward = total_asset - self.total_asset

        self.rewards.append(reward)
        self.total_asset = total_asset
        if done:
            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.episode_return = total_asset / self.initial_amount

        return state, reward, done, dict()

    def _sell_stock(self, index, action):
        adj_close_price = self.close_ary[self.day, index]  # `adjcp` denotes adjusted close price?

        if adj_close_price > 0:
            # Sell only if the price is > 0 (no missing data in this particular date)
            # perform sell action based on the sign of the action
            if self.stocks[index] > 0:
                # Sell only if current asset is > 0
                sell_num_shares0 = min(abs(action), self.stocks[index])
                sell_amount = adj_close_price * sell_num_shares0 * self.sell_cost_rate
                # update balance
                self.amount += sell_amount

                self.stocks[index] -= sell_num_shares0
            else:
                sell_num_shares0 = 0
        else:
            sell_num_shares0 = 0
        return sell_num_shares0

    def _buy_stock(self, index, action):
        adj_close_price = self.close_ary[self.day, index]  # `adjcp` denotes adjusted close price?

        if adj_close_price > 0:
            # Buy only if the price is > 0 (no missing data in this particular date)
            available_amount = self.amount // adj_close_price
            # print('available_amount:{}'.format(available_amount))

            # update balance
            buy_num_shares0 = min(available_amount, action)
            buy_amount = adj_close_price * buy_num_shares0 * self.buy_cost_rate
            self.amount -= buy_amount

            self.stocks[index] += buy_num_shares0

        else:
            buy_num_shares0 = 0
        return buy_num_shares0

    def get_close_ary_tech_ary(self, ticker_list=None, tech_id_list=None, beg_date=None, end_date=None, ):
        """source: https://github.com/AI4Finance-LLC/FinRL-Library
        finrl/autotrain/training.py
        finrl/preprocessing/preprocessing.py
        finrl/env/env_stocktrading.py
        """

        """hyper-parameters"""
        # cwd = './env/FinRL'
        cwd = "./" + config.DATA_SAVE_DIR
        ary_data_path = f'{cwd}/ary_data.npz'
        raw_data_path = f'{cwd}/raw_data.csv'
        prp_data_path = f'{cwd}/prp_data.csv'  # preprocessed data
        beg_date = '2008-03-19' if beg_date is None else beg_date
        end_date = '2021-01-01' if end_date is None else end_date
        ticker_list = ['AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP', 'HD',
                       'INTC', 'WMT', 'IBM', 'MRK', 'UNH', 'KO', 'CAT', 'TRV', 'JNJ', 'CVX', 'MCD',
                       'VZ', 'CSCO', 'XOM', 'BA', 'MMM', 'PFE', 'WBA', 'DD'
                       ] if ticker_list is None else ticker_list
        tech_id_list = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                        'close_30_sma', 'close_60_sma'
                        ] if tech_id_list is None else tech_id_list

        # """load from *.npz file"""
        # if not os.path.exists(ary_data_path):
        #     print(f"| FileNotFound: {ary_data_path}, so we download data from Internet.")
        #     print(f"  Can you download from github ↓ and put it in {ary_data_path}?")
        #     print(f"  https://github.com/Yonv1943/ElegantRL/env/FinRL/StockTradingEnv_ary_data.npz")
        #     print(f"  Or install finrl `pip3 install git+https://github.com/AI4Finance-LLC/FinRL-Library.git`")
        #     print(f"  Then it will download raw data {raw_data_path} from YaHoo")
        #
        #     os.makedirs(cwd, exist_ok=True)
        #     input(f'| If you have downloaded *.npz file or install finrl, press ENTER:')
        #
        # if os.path.exists(ary_data_path):
        #     ary_dict = np.load(ary_data_path, allow_pickle=True)
        #     close_ary = ary_dict['close_ary']
        #     tech_ary = ary_dict['tech_ary']
        #     return close_ary, tech_ary

        '''download and generate *.npz when FileNotFound'''
        print(f"| get_close_ary_tech_ary(), load: {raw_data_path}")
        df = self.raw_data_download(raw_data_path, beg_date, end_date, ticker_list[0])
        print(f"| get_close_ary_tech_ary(), load: {ary_data_path}")
        df = self.raw_data_preprocess(prp_data_path, df, beg_date, end_date, tech_id_list, )
        # import pandas as pd
        # df = pd.read_pickle(prp_data_path)  # DataFrame of Pandas

        # convert part of DataFrame to Numpy
        tech_ary = list()
        close_ary = list()
        df_len = len(df.index.unique())
        print(df_len)
        from tqdm import trange
        for day in trange(df_len):
            item = df.loc[day]

            # tech_items = [item[tech].values.tolist() for tech in tech_id_list]
            tech_items = [item[tech] for tech in tech_id_list]
            # tech_items_flatten = sum(tech_items, [])
            # tech_items_flatten = np.array(tech_items)
            tech_ary.append(tech_items)
            close_ary.append(item.close)

        close_ary = np.array(close_ary).reshape(-1, len(ticker_list))
        # data_ary = data_ary.reshape((-1, 5 * 30))
        tech_ary = np.array(tech_ary)
        print(f"| get_close_ary_tech_ary, close_ary.shape: {close_ary.shape}")
        print(f"| get_close_ary_tech_ary, tech_ary.shape: {tech_ary.shape}")
        np.savez_compressed(ary_data_path,
                            close_ary=np.array(close_ary),
                            tech_ary=np.array(tech_ary))
        return close_ary, tech_ary

    @staticmethod
    def raw_data_download(raw_data_path, beg_date, end_date, ticker_code):
        if os.path.exists(raw_data_path):
            import pandas as pd
            # raw_df = pd.read_pickle(raw_data_path)  # DataFrame of Pandas
            raw_df = pd.read_csv(raw_data_path)  # DataFrame of Pandas
            print('| raw_df.columns.values:', raw_df.columns.values)
            # ['date' 'open' 'high' 'low' 'close' 'volume' 'tic' 'day']
        else:
            import pandas as pd
            # 下载A股的日K线数据
            stock_data = StockData(output_dir="./" + config.DATA_SAVE_DIR, date_start=beg_date, date_end=end_date)
            # 获得数据文件路径
            csv_file_path = stock_data.download(ticker_code, fields=stock_data.fields_day)
            raw_df = pd.read_csv(csv_file_path)
            # raw_df.to_csv(ticker_code)
            # from finrl.marketdata.yahoodownloader import YahooDownloader
            # yd = YahooDownloader(start_date=beg_date, end_date=end_date, ticker_list=ticker_list, )
            # raw_df = yd.fetch_data()
            # raw_df.to_pickle(raw_data_path)
        return raw_df

    @staticmethod
    def raw_data_preprocess(prp_data_path, df, beg_date, end_date, tech_id_list, ):
        if os.path.exists(prp_data_path):
            import pandas as pd
            # df = pd.read_pickle(prp_data_path)  # DataFrame of Pandas
            df = pd.read_csv(prp_data_path)  # DataFrame of Pandas
        else:
            # from finrl.preprocessing.preprocessors import FeatureEngineer
            from FinRL_Library_master.finrl.preprocessing.preprocessors import FeatureEngineer
            fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=tech_id_list,
                                 use_turbulence=False, user_defined_feature=False, )
            df = fe.preprocess_data(df)  # preprocess raw_df

            df = df[(df.date >= beg_date) & (df.date < end_date)]
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]

            # df.to_pickle(prp_data_path)
            df.to_csv(prp_data_path)

        print('| df.columns.values:', df.columns.values)
        # assert all(df.columns.values == [
        #     'date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day', 'macd',
        #     'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma',
        #     'close_60_sma', 'turbulence'])
        return df

# def check_finrl_env():
#     from finrl.config import config
#     from numpy import random as rd
#
#     env_kwargs = {
#         "max_stock": 100,
#         "initial_amount": 1000000,
#         "buy_cost_pct": 0.001,
#         "sell_cost_pct": 0.001,
#         # "state_space": 1 + (2 + len(config.TECHNICAL_INDICATORS_LIST)) * stock_dimension,
#         "stock_dim": len(config.DOW_30_TICKER),
#         "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
#         # "action_space": stock_dimension,
#         "reward_scaling": 1e-4
#     }
#
#     env = StockTradingEnv(**env_kwargs)
#     action_dim = len(config.DOW_30_TICKER)
#
#     state = env.reset()
#     print('state_dim', len(state))
#
#     done = False
#     step = 1
#     from time import time
#     timer = time()
#     while not done:
#         action = rd.rand(action_dim) * 2 - 1
#         next_state, reward, done, _ = env.step(action)
#         # print(';', step, len(next_state), env.day, reward)
#         step += 1
#
#     print(f"step: {step}, UsedTime: {time() - timer:.3f}")  # 44 seconds
#
#
# if __name__ == '__main__':
#     # check_finance_stock_env()
#     check_finrl_env()
