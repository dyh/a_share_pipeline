import sys
from os.path import dirname, abspath

from pipeline.sqlite import SQLite

sys.path.append('../FinRL_Library_master')

import baostock as bs
import pandas as pd
import os
from FinRL_Library_master.finrl.config import config
import numpy as np

from FinRL_Library_master.finrl.preprocessing.preprocessors import FeatureEngineer

# 5分钟K线的时间点
list_time_point_5minutes = ['09:35:00',
                            '09:40:00',
                            '09:45:00',
                            '09:50:00',
                            '09:55:00',
                            '10:00:00',
                            '10:05:00',
                            '10:10:00',
                            '10:15:00',
                            '10:20:00',
                            '10:25:00',
                            '10:30:00',
                            '10:35:00',
                            '10:40:00',
                            '10:45:00',
                            '10:50:00',
                            '10:55:00',
                            '11:00:00',
                            '11:05:00',
                            '11:10:00',
                            '11:15:00',
                            '11:20:00',
                            '11:25:00',
                            '11:30:00',
                            '13:05:00',
                            '13:10:00',
                            '13:15:00',
                            '13:20:00',
                            '13:25:00',
                            '13:30:00',
                            '13:35:00',
                            '13:40:00',
                            '13:45:00',
                            '13:50:00',
                            '13:55:00',
                            '14:00:00',
                            '14:05:00',
                            '14:10:00',
                            '14:15:00',
                            '14:20:00',
                            '14:25:00',
                            '14:30:00',
                            '14:35:00',
                            '14:40:00',
                            '14:45:00',
                            '14:50:00',
                            '14:55:00',
                            '15:00:00']

# 15分钟K线的时间点
list_time_point_15minutes = ['09:45:00',
                             '10:00:00',
                             '10:15:00',
                             '10:30:00',
                             '10:45:00',
                             '11:00:00',
                             '11:15:00',
                             '11:30:00',
                             '13:15:00',
                             '13:30:00',
                             '13:45:00',
                             '14:00:00',
                             '14:15:00',
                             '14:30:00',
                             '14:45:00',
                             '15:00:00']


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class StockData(object):
    def __init__(self, output_dir, date_start, date_end):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        self.date_end = date_end
        self.output_dir = output_dir

        # self.fields_day = "date,open,high,low,close,volume,code,amount,turn," \
        #                   "tradestatus,pctChg,peTTM, pbMRQ,psTTM,pcfNcfTTM"

        self.fields_day = "date,open,high,low,close,volume,code"

        self.fields_minutes = "time,open,high,low,volume,amount,code,close"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def download_raw_data(self, code_list, fields, frequency='d', adjustflag='3'):
        df_result = pd.DataFrame()

        for code in code_list:

            df_temp = bs.query_history_k_data_plus(code, fields=fields,
                                                   start_date=self.date_start,
                                                   end_date=self.date_end,
                                                   frequency=frequency,
                                                   adjustflag=adjustflag).get_data()

            # 删除停牌日期的行，即 tradestatus=0 的数据，只保留 tradestatus=1 的数据
            if 'tradestatus' in df_temp.columns:
                df_temp = df_temp[df_temp['tradestatus'] == '1']
                # 删除 tradestatus 列、adjustflag 列、isST 列
                df_temp.drop(['tradestatus'], axis=1, inplace=True)
                # df.drop(['tradestatus', 'adjustflag', 'isST'], axis=1, inplace=True)

            # 更改列名
            if 'code' in df_temp.columns:
                df_temp = df_temp.rename(columns={'code': 'tic'})

            # 替换空格和回车为nan
            # df.replace(to_replace=' ', value=np.nan, inplace=True)
            df_temp.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

            # 將nan填充为0
            df_temp.fillna(0, inplace=True)
            # 统计nan值
            # print(df.isnull().sum())
            df_result = df_result.append(df_temp)
        pass

        # csv_file_path = f'{self.output_dir}/{csv_file_name}'
        # df_result.to_csv(csv_file_path, index=False)
        self.exit()

        df_result['open'] = df_result['open'].astype(np.float32)
        df_result['high'] = df_result['high'].astype(np.float32)
        df_result['low'] = df_result['low'].astype(np.float32)
        df_result['close'] = df_result['close'].astype(np.float32)
        df_result['volume'] = df_result['volume'].astype(np.int)

        return df_result

    def download(self, code, fields, frequency='d', adjustflag='3'):
        """

        :param fields:
        :param code:
        :param frequency: 数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写
        :param adjustflag: 复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权
        :return:
        """
        df = bs.query_history_k_data_plus(code, fields=fields,
                                          start_date=self.date_start,
                                          end_date=self.date_end, frequency=frequency, adjustflag=adjustflag).get_data()

        # 删除停牌日期的行，即 tradestatus=0 的数据，只保留 tradestatus=1 的数据

        if 'tradestatus' in df.columns:
            df = df[df['tradestatus'] == '1']
            # 删除 tradestatus 列、adjustflag 列、isST 列
            df.drop(['tradestatus'], axis=1, inplace=True)
            # df.drop(['tradestatus', 'adjustflag', 'isST'], axis=1, inplace=True)

        # 更改列名
        if 'code' in df.columns:
            df = df.rename(columns={'code': 'tic'})

        # 替换空格和回车为nan
        # df.replace(to_replace=' ', value=np.nan, inplace=True)
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

        # 將nan填充为0
        df.fillna(0, inplace=True)
        # 统计nan值
        # print(df.isnull().sum())

        csv_file_path = f'{self.output_dir}/{code}.csv'
        df.to_csv(csv_file_path, index=False)

        # 下载所有股票数据
        # stock_df = self.get_codes_by_date(self.date_end)
        # for index, row in stock_df.iterrows():
        #     print(f'processing {row["code"]} {row["code_name"]}')

        self.exit()

        return csv_file_path

    def get_informer_data(self, stock_code, fields, frequency, adjustflag, output_file_name='ETTm1.csv',
                          use_technical_indicator=False):
        """
        获取 informer 格式数据
        :param stock_code: A股代码 sh.600036
        :param fields: 字段 self.fields_minutes
        :param frequency: 数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写
        :param adjustflag: 复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权
        :param output_file_name: 输出的文件名
        :param use_technical_indicator: 是否增加技术指标列
        :return: csv 文件路径
        """

        # 股票代码
        stock_code = stock_code

        # 训练开始日期
        start_date = self.date_start

        # 停止预测日期
        end_date = self.date_end

        print("==============下载A股数据==============")

        # 下载A股的日K线数据
        stock_data = StockData(self.output_dir, date_start=start_date, date_end=end_date)

        # 获得数据文件路径
        csv_file_path = stock_data.download(stock_code, fields=fields, frequency=frequency, adjustflag=adjustflag)

        df = pd.read_csv(csv_file_path)

        if use_technical_indicator is True:
            print("==============加入技术指标==============")
            fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                use_turbulence=False,
                user_defined_feature=False,
            )

            df = fe.preprocess_data(df)
            # df_fe['log_volume'] = np.log(df_fe.volume * df_fe.close)
            df['change'] = (df.close - df.open) / df.close
            df['daily_variance'] = (df.high - df.low) / df.close
        else:
            df = df
        pass

        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

        # 將nan填充为0
        df.fillna(0, inplace=True)

        df['time'] = df['time'].astype('str').str[:-3]
        df["time"] = pd.to_datetime(df.time)
        # 将time列，改名为date
        df = df.rename(columns={'time': 'date'})

        # 将 close 移动到最后一列，改名为OT
        col_close = df.close
        df = df.drop('close', axis=1)
        df['OT'] = col_close

        if 'tic' in df.columns:
            df = df.drop('tic', axis=1)

        # df_fe.to_csv(f'{self.output_dir}/{stock_code}_processed_df.csv', index=False)
        df.to_csv(f'{self.output_dir}/{output_file_name}', index=False)

        print("==============数据准备完成==============")
        pass

    @staticmethod
    def create_a_share_tables():
        # 获取沪深300股信息
        list_hs_300 = StockData.get_hs300_stocks()

        # 连接数据库
        sqlite = SQLite(dbname="./elegant/" + config.DATA_SAVE_DIR + '/a_share.db')

        for stock_item in list_hs_300:
            stock_date = stock_item[0]
            stock_code = stock_item[1]
            stock_name = stock_item[2]

            # 查询是否有同名的表
            if_exists = sqlite.table_exists(stock_code)
            # None
            print(if_exists)

            # 如果没有，则新建
            if if_exists is None:

                sqlite.execute_non_query(sql=f'CREATE TABLE "{stock_code}" (ID INTEGER PRIMARY KEY AUTOINCREMENT, '
                                             f'DATE TEXT NOT NULL, OPEN TEXT NOT NULL,'
                                             f'HIGH TEXT, LOW TEXT, CLOSE TEXT);')

                pass
            pass

        sqlite.close()

        pass

    @staticmethod
    def get_hs300_stocks():
        """
        获取沪深300股的代码列表
        :return: list[]
        """
        bs.login()
        rs = bs.query_hs300_stocks()
        hs300_stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            hs300_stocks.append(rs.get_row_data())
        pass
        # result = pd.DataFrame(hs300_stocks, columns=rs.fields)
        bs.logout()
        return hs300_stocks

    @staticmethod
    def fill_zero_value_to_null_date(df, code_list, date_column_name='date', code_column_name='tic'):
        """
        向没有数据的日期填充 0 值
        :param code_column_name: 股票代码列的名称
        :param date_column_name: 日期列的名称
        :param code_list: 股票代码列表
        :param df: DataFrame数据
        :return: df
        """

        # 读取 date 列的数据，去重复，形成 date_list
        column_date = df[date_column_name].unique()

        list_date = list(column_date)

        # 遍历 date_list
        for item_date in list_date:
            # 根据 code_list ，找到某些天，没有 数据的 code
            for item_code in code_list:
                # df[(df['a'] > 0) & (df['b'] < 0) | df['c'] > 0]
                item_temp = df[(df[date_column_name] == item_date) & (df[code_column_name] == item_code)]
                # 如果查询结果为空
                if item_temp.empty:
                    # 用 code 作为code_column_name的值，创建一条新的 df 数据，append进df
                    # 增加一条以 0 填充的空白记录
                    new_row = pd.DataFrame({code_column_name: str(item_code), date_column_name: str(item_date)},
                                           index=[0])

                    df = df.append(new_row, ignore_index=True)
                    pass
                pass
            pass
        pass

        # 用0填充空白
        df.fillna(0, inplace=True)

        # 重新索引和排序
        df = df.sort_values(by=[date_column_name, code_column_name]).reset_index(drop=True)

        return df
        pass

if __name__ == '__main__':
    # list_300 = StockData.get_hs300_stocks()
    # list_300[0][1]
    # 'sh.600000'
    # print(list_300)

    StockData.create_a_share_tables()

    print('1')
    #

    pass
