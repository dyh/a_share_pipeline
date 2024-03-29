import time

import pandas as pd
import numpy as np

import config
import train_helper
from train_helper import insert_train_history_record_sqlite


class StockTradingEnvBatch:
    def __init__(self, cwd='./envs/FinRL', gamma=0.99,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, sell_cost_pct=1e-3,
                 start_date='2008-03-19', end_date='2016-01-01', env_eval_date='2021-01-01',
                 ticker_list=None, tech_indicator_list=None, initial_stocks=None, if_eval=False,
                 fe_table_name=''):

        self.price_ary, self.tech_ary, self.tic_ary, self.date_ary = self.load_data(fe_table_name, if_eval, ticker_list,
                                                                                    tech_indicator_list,
                                                                                    start_date, end_date,
                                                                                    env_eval_date)
        stock_dim = self.price_ary.shape[1]

        self.gamma = gamma
        self.max_stock = max_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_capital = initial_capital
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.initial_total_asset = None
        self.gamma_reward = 0.0

        # environment information
        self.env_name = 'StockTradingEnv-v1'
        self.state_dim = 1 + 2 * stock_dim + self.tech_ary.shape[1]
        self.action_dim = stock_dim
        self.max_step = len(self.price_ary) - 1
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        # 奖励 比例
        self.reward_scale = 0.0

        self.state_amount_scale = 0.0
        self.state_price_scale = 0.0
        self.state_stocks_scale = 0.0
        self.state_tech_scale = 0.0

        # 是 eval 还是 train
        self.if_eval = if_eval

        # 为每一只股票分配 独立 的输出的list
        self.list_buy_or_sell_output = []

        # 为每一只股票分配 独立 的现金
        self.amount_ary = []
        # 初始化变量
        for index in range(self.action_dim):
            self.amount_ary.append(0)
            self.list_buy_or_sell_output.append([])
            pass
        pass

    def get_trade_detail(self, yesterday_price, index, price, date_temp, tic_temp, actions, sell_num_shares,
                         buy_num_shares):
        if yesterday_price[index] != 0:
            price_diff_percent = str(
                round((price[index] - yesterday_price[index]) / yesterday_price[index], 4))
        else:
            price_diff_percent = '0.0'
        pass

        price_diff = str(round(price[index] - yesterday_price[index], 6))

        asset_temp = self.amount_ary[index] + (self.stocks[index] * price[index])

        trade_detail_temp = f'第 {self.day + 1} 天，{date_temp}\r\n'
        trade_detail_temp += f'        > {tic_temp}，预测涨跌：{round(-1 * actions[index], 4)}，' \
                             f'实际涨跌：{price_diff_percent} ￥{price_diff} 元，' \
                             f'买{buy_num_shares} 卖{sell_num_shares} 股, 持股数量 {self.stocks[index]}，' \
                             f'现金：{self.amount_ary[index]}，资产：{asset_temp}\r\n'

        return trade_detail_temp

    def reset(self):

        # 如果是正式预测，输出到网页，固定 持股数和现金
        if config.IF_ACTUAL_PREDICT is True:
            # 日期 固定
            self.day = 0
            price = self.price_ary[self.day]

            self.stocks = self.initial_stocks.copy()
            self.amount = self.initial_capital - (self.stocks * price).sum()
        else:
            # 随机 日期 起点
            np.random.seed(round(time.time()))
            self.day = np.random.randint(0, self.max_step * 0.5)

            price = self.price_ary[self.day]

            # 如果是train过程中的eval
            np.random.seed(round(time.time()))
            random_float = np.random.uniform(0.0, 1.01, size=self.initial_stocks.shape)

            self.stocks = random_float * self.initial_stocks.copy() // config.\
                MINIMUM_TRADE_SHARES * config.MINIMUM_TRADE_SHARES

            # 现金 金额
            cash_temp = self.initial_capital * np.random.uniform(0.95, 1.05)
            # 股票 金额
            share_temp = (self.stocks * price).sum()

            # 如果股票金额比现金多，则再次random，直到现金比股票多
            while share_temp > cash_temp:
                np.random.seed(round(time.time()))
                random_float = np.random.uniform(0.0, 1.01, size=self.initial_stocks.shape)

                self.stocks = random_float * self.initial_stocks.copy() // config. \
                    MINIMUM_TRADE_SHARES * config.MINIMUM_TRADE_SHARES

                # np.random.seed(round(time.time()))
                cash_temp = self.initial_capital * np.random.uniform(0.95, 1.05)
                share_temp = (self.stocks * price).sum()
                pass
            pass

            self.amount = cash_temp - share_temp

        pass

        for index in range(self.action_dim):
            # 平均分 现金
            self.amount_ary[index] = self.amount / self.action_dim

            # 独立的输出的list
            self.list_buy_or_sell_output[index] = []
        pass

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        state = np.hstack((self.amount * 2 ** self.state_amount_scale,
                           price * 2 ** self.state_price_scale,
                           self.stocks * 2 ** self.state_stocks_scale,
                           self.tech_ary[self.day] * 2 ** self.state_tech_scale,)).astype(np.float32)

        if config.IF_DEBUG_STATE_SCALE is True:
            max_state_value = np.max(state)

            if max_state_value >= 1.0:
                # if max_state_value is not None:
                amount_temp = float(np.max(self.amount * 2 ** self.state_amount_scale))
                price_temp = float(np.max(price * 2 ** self.state_price_scale))
                stocks_temp = float(np.max(self.stocks * 2 ** self.state_stocks_scale))
                tech_temp = float(np.max(self.tech_ary[self.day] * 2 ** self.state_tech_scale))

                insert_train_history_record_sqlite(model_id=config.MODEL_HYPER_PARAMETERS,
                                                   state_amount_value=amount_temp, state_price_value=price_temp,
                                                   state_stocks_value=stocks_temp, state_tech_value=tech_temp)

                print('>>>> reset() -> if_eval:', self.if_eval, 'max_state_value >= 1 | amount:', amount_temp,
                      '| stocks', stocks_temp, '| tech', tech_temp, '| price', price_temp)

                pass
            pass
        pass

        return state

    def step(self, actions):
        int_type_actions = (actions * self.max_stock).astype(int)

        # ----
        yesterday_price = self.price_ary[self.day]
        # ----

        self.day += 1

        price = self.price_ary[self.day]

        tic_ary_temp = self.tic_ary[self.day]
        # 日期
        date_ary_temp = self.date_ary[self.day]
        date_temp = date_ary_temp[0]

        for index in np.where(int_type_actions < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0

                sell_num_shares = min(self.stocks[index], -int_type_actions[index])

                tic_temp = tic_ary_temp[index]

                if sell_num_shares >= config.MINIMUM_TRADE_SHARES:
                    # 若 action <= -100 地板除，卖1手整
                    sell_num_shares = sell_num_shares // config.MINIMUM_TRADE_SHARES * config.MINIMUM_TRADE_SHARES
                    self.stocks[index] -= sell_num_shares
                    self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

                    # 平均现金
                    self.amount_ary[index] += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

                    if config.IF_ACTUAL_PREDICT is True:
                        # tic, date, sell/buy, hold, 第x天
                        # episode_return_temp = (self.amount_ary[index] + (
                        #         self.stocks * price).sum()) / self.initial_total_asset

                        episode_return_temp = (self.amount_ary[index] + self.stocks[index] * price[index]) / \
                                              (self.initial_total_asset / self.action_dim)

                        # 获取交易详情
                        trade_detail_temp = self.get_trade_detail(yesterday_price, index, price, date_temp, tic_temp,
                                                                  actions, sell_num_shares=sell_num_shares,
                                                                  buy_num_shares=0)

                        list_item = (tic_temp, date_temp, -1 * sell_num_shares, self.stocks[index], self.day + 1,
                                     episode_return_temp, trade_detail_temp)

                        # 添加到输出list
                        self.list_buy_or_sell_output[index].append(list_item)
                    pass
                else:
                    # 当sell_num_shares < 100时，判断若 self.stocks[index] >= 100 则放大效果，卖1手
                    if self.stocks[index] >= config.MINIMUM_TRADE_SHARES:
                        sell_num_shares = config.MINIMUM_TRADE_SHARES
                        self.stocks[index] -= sell_num_shares
                        self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

                        self.amount_ary[index] += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

                        if config.IF_ACTUAL_PREDICT is True:
                            # tic, date, sell/buy, hold, 第x天
                            # episode_return_temp = (self.amount_ary[index] + (
                            #         self.stocks * price).sum()) / self.initial_total_asset

                            episode_return_temp = (self.amount_ary[index] + self.stocks[index] * price[index]) / \
                                                  (self.initial_total_asset / self.action_dim)

                            # 获取交易详情
                            trade_detail_temp = self.get_trade_detail(yesterday_price, index, price, date_temp,
                                                                      tic_temp, actions,
                                                                      sell_num_shares=sell_num_shares, buy_num_shares=0)

                            list_item = (tic_temp, date_temp, -1 * sell_num_shares, self.stocks[index], self.day + 1,
                                         episode_return_temp, trade_detail_temp)
                            # 添加到输出list
                            self.list_buy_or_sell_output[index].append(list_item)
                        pass
                    else:
                        # self.stocks[index] 不足1手时，不动
                        sell_num_shares = 0

                        if config.IF_ACTUAL_PREDICT is True:
                            # tic, date, sell/buy, hold, 第x天
                            # episode_return_temp = (self.amount_ary[index] + (
                            #         self.stocks * price).sum()) / self.initial_total_asset

                            episode_return_temp = (self.amount_ary[index] + self.stocks[index] * price[index]) / \
                                                  (self.initial_total_asset / self.action_dim)

                            # 获取交易详情
                            trade_detail_temp = self.get_trade_detail(yesterday_price, index, price, date_temp,
                                                                      tic_temp, actions,
                                                                      sell_num_shares=sell_num_shares,
                                                                      buy_num_shares=0)

                            list_item = (tic_temp, date_temp, 0, self.stocks[index], self.day + 1,
                                         episode_return_temp, trade_detail_temp)
                            # 添加到输出list
                            self.list_buy_or_sell_output[index].append(list_item)
                        pass
                    pass
                pass

            pass
        pass

        for index in np.where(int_type_actions > 0)[0]:  # buy_index:
            if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                # 为每只股票单独分配现金， self.amount 除以 股票只数, stock_dim
                buy_num_shares = min(self.amount_ary[index] / (1 + self.buy_cost_pct) // price[index], int_type_actions[index])

                tic_temp = tic_ary_temp[index]

                if buy_num_shares >= config.MINIMUM_TRADE_SHARES:
                    # 若 actions >= +100，地板除，买1手整
                    buy_num_shares = buy_num_shares // config.MINIMUM_TRADE_SHARES * config.MINIMUM_TRADE_SHARES
                    self.stocks[index] += buy_num_shares
                    # 手续费
                    self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

                    self.amount_ary[index] -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

                    if config.IF_ACTUAL_PREDICT is True:
                        # tic, date, sell/buy, hold, 第x天
                        # episode_return_temp = (self.amount_ary[index] + (
                        #         self.stocks * price).sum()) / self.initial_total_asset

                        episode_return_temp = (self.amount_ary[index] + self.stocks[index] * price[index]) / \
                                              (self.initial_total_asset / self.action_dim)

                        # 获取交易详情
                        trade_detail_temp = self.get_trade_detail(yesterday_price, index, price, date_temp,
                                                                  tic_temp,
                                                                  actions, sell_num_shares=0,
                                                                  buy_num_shares=buy_num_shares)

                        list_item = (tic_temp, date_temp, buy_num_shares, self.stocks[index], self.day + 1,
                                     episode_return_temp, trade_detail_temp)

                        # 添加到输出list
                        self.list_buy_or_sell_output[index].append(list_item)
                    pass
                else:
                    # 当buy_num_shares < 100时，判断若 self.amount // price[index] >= 100，则放大效果，买1手
                    if (self.amount_ary[index] / (1 + self.buy_cost_pct) // price[index]) >= config.MINIMUM_TRADE_SHARES:

                        buy_num_shares = config.MINIMUM_TRADE_SHARES
                        self.stocks[index] += buy_num_shares
                        self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

                        self.amount_ary[index] -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

                        if config.IF_ACTUAL_PREDICT is True:
                            # tic, date, sell/buy, hold, 第x天
                            # episode_return_temp = (self.amount_ary[index] + (
                            #         self.stocks * price).sum()) / self.initial_total_asset

                            episode_return_temp = (self.amount_ary[index] + self.stocks[index] * price[index]) / \
                                                  (self.initial_total_asset / self.action_dim)

                            # 获取交易详情
                            trade_detail_temp = self.get_trade_detail(yesterday_price, index, price, date_temp,
                                                                      tic_temp,
                                                                      actions, sell_num_shares=0,
                                                                      buy_num_shares=buy_num_shares)

                            list_item = (tic_temp, date_temp, buy_num_shares, self.stocks[index], self.day + 1,
                                         episode_return_temp, trade_detail_temp)

                            # 添加到输出list
                            self.list_buy_or_sell_output[index].append(list_item)
                    else:
                        # self.amount // price[index] 不足100时，不动
                        # 未达到1手，不买
                        buy_num_shares = 0

                        if config.IF_ACTUAL_PREDICT is True:
                            # tic, date, sell/buy, hold, 第x天
                            # episode_return_temp = (self.amount_ary[index] + (
                            #         self.stocks * price).sum()) / self.initial_total_asset

                            episode_return_temp = (self.amount_ary[index] + self.stocks[index] * price[index]) / \
                                                  (self.initial_total_asset / self.action_dim)

                            # 获取交易详情
                            trade_detail_temp = self.get_trade_detail(yesterday_price, index, price, date_temp,
                                                                      tic_temp,
                                                                      actions, sell_num_shares=0,
                                                                      buy_num_shares=buy_num_shares)

                            list_item = (tic_temp, date_temp, 0, self.stocks[index], self.day + 1, episode_return_temp,
                                         trade_detail_temp)

                            # 添加到输出list
                            self.list_buy_or_sell_output[index].append(list_item)
                        pass
                    pass
                pass

            pass
        pass

        if config.IF_ACTUAL_PREDICT is True:
            for index in np.where(int_type_actions == 0)[0]:  # action=0
                if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                    # tic, date, sell/buy, hold, 第x天
                    tic_temp = tic_ary_temp[index]
                    # episode_return_temp = (self.amount_ary[index] + (
                    #         self.stocks * price).sum()) / self.initial_total_asset

                    episode_return_temp = (self.amount_ary[index] + self.stocks[index] * price[index]) / \
                                          (self.initial_total_asset / self.action_dim)

                    # 获取交易详情
                    trade_detail_temp = self.get_trade_detail(yesterday_price, index, price, date_temp, tic_temp,
                                                              actions, 0, 0)

                    list_item = (tic_temp, date_temp, 0, self.stocks[index], self.day + 1, episode_return_temp,
                                 trade_detail_temp)
                    # 添加到输出list
                    self.list_buy_or_sell_output[index].append(list_item)
                    pass
                pass
            pass
        pass

        state = np.hstack((self.amount * 2 ** self.state_amount_scale,
                           price * 2 ** self.state_price_scale,
                           self.stocks * 2 ** self.state_stocks_scale,
                           self.tech_ary[self.day] * 2 ** self.state_tech_scale,)).astype(np.float32)

        if config.IF_DEBUG_STATE_SCALE is True:
            max_state_value = np.max(state)

            if max_state_value >= 1.0:
                amount_temp = float(np.max(self.amount * 2 ** self.state_amount_scale))
                price_temp = float(np.max(price * 2 ** self.state_price_scale))
                stocks_temp = float(np.max(self.stocks * 2 ** self.state_stocks_scale))
                tech_temp = float(np.max(self.tech_ary[self.day] * 2 ** self.state_tech_scale))

                insert_train_history_record_sqlite(model_id=config.MODEL_HYPER_PARAMETERS,
                                                   state_amount_value=amount_temp, state_price_value=price_temp,
                                                   state_stocks_value=stocks_temp, state_tech_value=tech_temp)

                print('>>>> step() -> if_eval:', self.if_eval, 'max_state_value >= 1: | amount:', amount_temp,
                      '| stocks', stocks_temp, '| tech', tech_temp, '| price', price_temp)
            pass
        pass

        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * 2 ** self.reward_scale

        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step

        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        else:

            if config.IF_DEBUG_REWARD_SCALE is True:
                if reward > config.REWARD_THRESHOLD:
                    # 如果是 预测
                    if self.if_eval is True:
                        insert_train_history_record_sqlite(model_id=config.MODEL_HYPER_PARAMETERS,
                                                           eval_reward_value=reward)

                        print('>>>>', config.AGENT_NAME, 'eval reward', str(reward))
                    else:
                        # 如果是 train
                        insert_train_history_record_sqlite(model_id=config.MODEL_HYPER_PARAMETERS,
                                                           train_reward_value=reward)

                        print('>>>>', config.AGENT_NAME, 'train reward', str(reward))
                    pass
                pass
            pass
        pass

        return state, reward, done, dict()

    def load_data(self, table_name='', if_eval=None,
                  ticker_list=None, tech_indicator_list=None,
                  start_date='2008-03-19', end_date='2016-01-01', env_eval_date='2021-01-01'):

        # 从数据库中读取fe fillzero的数据
        from stock_data import StockData
        processed_df = StockData.get_fe_fillzero_from_sqlite(begin_date=start_date, end_date=env_eval_date,
                                                             list_stock_code=ticker_list,
                                                             table_name=table_name,
                                                             if_actual_predict=config.IF_ACTUAL_PREDICT)

        def data_split_train(df, start, end):
            data = df[(df.date >= start) & (df.date < end)]
            data = data.sort_values(["date", "tic"], ignore_index=True)
            data.index = data.date.factorize()[0]
            return data

        def data_split_eval(df, start, end):
            data = df[(df.date >= start) & (df.date <= end)]
            data = data.sort_values(["date", "tic"], ignore_index=True)
            data.index = data.date.factorize()[0]
            return data

        train_df = data_split_train(processed_df, start_date, end_date)
        eval_df = data_split_eval(processed_df, end_date, env_eval_date)

        train_price_ary, train_tech_ary, train_tic_ary, train_date_ary = self.convert_df_to_ary(train_df,
                                                                                                tech_indicator_list)
        eval_price_ary, eval_tech_ary, eval_tic_ary, eval_date_ary = self.convert_df_to_ary(eval_df,
                                                                                            tech_indicator_list)

        if if_eval is None:
            price_ary = np.concatenate((train_price_ary, eval_price_ary), axis=0)
            tech_ary = np.concatenate((train_tech_ary, eval_tech_ary), axis=0)
            tic_ary = None
            date_ary = None
        elif if_eval:
            price_ary = eval_price_ary
            tech_ary = eval_tech_ary
            tic_ary = eval_tic_ary
            date_ary = eval_date_ary
        else:
            price_ary = train_price_ary
            tech_ary = train_tech_ary
            tic_ary = train_tic_ary
            date_ary = train_date_ary

        return price_ary, tech_ary, tic_ary, date_ary

    @staticmethod
    def convert_df_to_ary(df, tech_indicator_list):
        tech_ary = list()
        price_ary = list()
        tic_ary = list()
        date_ary = list()

        from stock_data import fields_prep
        columns_list = fields_prep.split(',')

        for day in range(len(df.index.unique())):
            # item = df.loc[day]
            list_temp = df.loc[day]
            if list_temp.ndim == 1:
                list_temp = [df.loc[day]]
                pass
            item = pd.DataFrame(data=list_temp, columns=columns_list)

            tech_items = [item[tech].values.tolist() for tech in tech_indicator_list]
            tech_items_flatten = sum(tech_items, [])
            tech_ary.append(tech_items_flatten)
            price_ary.append(item.close)  # adjusted close price (adjcp)

            # ----
            # tic_ary.append(list(item.tic))
            # date_ary.append(list(item.date))

            tic_ary.append(item.tic)
            date_ary.append(item.date)

            # ----

            pass

        price_ary = np.array(price_ary)
        tech_ary = np.array(tech_ary)

        tic_ary = np.array(tic_ary)
        date_ary = np.array(date_ary)

        print(f'| price_ary.shape: {price_ary.shape}, tech_ary.shape: {tech_ary.shape}')
        return price_ary, tech_ary, tic_ary, date_ary

    def draw_cumulative_return(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(self.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)
                action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                total_asset = self.amount + (self.price_ary[self.day] * self.stocks).sum()
                episode_return = total_asset / self.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break

        import matplotlib.pyplot as plt
        plt.plot(episode_returns)
        plt.grid()
        plt.title('cumulative return')
        plt.xlabel('day')
        plt.xlabel('multiple of initial_account')
        plt.savefig(f'{cwd}/cumulative_return.jpg')
        return episode_returns


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data
    from finrl.preprocessing.preprocessors import FeatureEngineer

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
            self,
            use_technical_indicator=True,
            tech_indicator_list=None,  # config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=False,
            user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        if self.use_technical_indicator:
            # add technical indicators using stockstats
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        print('df.fillna ...')
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        from stockstats import StockDataFrame as Sdf  # for Sdf.retype

        df = data.copy()
        df = df.sort_values(by=['tic', 'date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        count_tech_indicator = len(self.tech_indicator_list)
        count_ticker = len(unique_ticker)

        tech_indicator_index = 0

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )

                    print('indicator', tech_indicator_index + 1, '/', count_tech_indicator,
                          indicator, i + 1, '/', count_ticker)

                except Exception as e:
                    print(e)

            tech_indicator_index += 1
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')

        df = df.sort_values(by=['date', 'tic'])
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    @staticmethod
    def add_user_defined_feature(data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    @staticmethod
    def calculate_turbulence(data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index
