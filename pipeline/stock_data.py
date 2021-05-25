import sys

from pipeline.utils.psqldb import Psqldb

sys.path.append('../FinRL_Library_master')

from pipeline.sqlite import SQLite
from pipeline.utils.datetime import get_today_date, is_greater, get_datetime_from_date_str, get_next_work_day, \
    get_next_day

import baostock as bs
import pandas as pd
import os
from pipeline.elegant import config

import numpy as np

import requests

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

fields_day = "date,open,high,low,close,volume,code"

fields_prep = 'date,open,high,low,close,volume,tic,macd,boll_ub,boll_lb,rsi_30,cci_30,dx_30,close_30_sma,close_60_sma'

fields_minutes = "time,open,high,low,volume,amount,code,close"


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class StockData(object):
    def __init__(self):
        self._bs = bs
        bs.login()
        self.date_start = ''
        self.date_end = ''
        self.output_dir = ''

        # self.fields_day = "date,open,high,low,close,volume,code,amount,turn," \
        #                   "tradestatus,pctChg,peTTM, pbMRQ,psTTM,pcfNcfTTM"

        # self.fields_day = "date,open,high,low,close,volume,code"
        # self.fields_minutes = "time,open,high,low,volume,amount,code,close"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def download_raw_data(self, code_list, fields, date_start='', date_end='', frequency='d', adjustflag='3'):
        self.date_start = date_start
        self.date_end = date_end

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

        if df_result.empty is True:
            return None

        # csv_file_path = f'{self.output_dir}/{csv_file_name}'
        # df_result.to_csv(csv_file_path, index=False)

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
    def create_fe_table(db_path, table_name):
        # 连接数据库
        sqlite = SQLite(db_path)

        if_exists = sqlite.table_exists(table_name)

        if if_exists is None:
            # 如果是初始化，则创建表
            sqlite.execute_non_query(sql=f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                         f'date TEXT NOT NULL, open TEXT NOT NULL, '
                                         f'high TEXT NOT NULL, low TEXT NOT NULL, '
                                         f'close TEXT NOT NULL, volume TEXT NOT NULL, '
                                         f'tic TEXT NOT NULL, macd TEXT NOT NULL, '
                                         f'boll_ub TEXT NOT NULL, boll_lb TEXT NOT NULL, '
                                         f'rsi_30 TEXT NOT NULL, cci_30 TEXT NOT NULL, '
                                         f'dx_30 TEXT NOT NULL, close_30_sma TEXT NOT NULL, '
                                         f'close_60_sma TEXT NOT NULL'
                                         f');')
            # 提交
            sqlite.commit()
            sqlite.close()
            pass
        pass

    @staticmethod
    def load_hs300_from_sqlite(db_path, list_hs_300, date_begin='', date_end=''):
        """
        从sqlite数据库获取沪深300股信息
        :param db_path: 数据库地址
        :param list_hs_300: 300个代码
        :param date_begin: 开始日期
        :param date_end: 结束日期
        :return: DataFrame
        """
        df = pd.DataFrame(data=None, columns=fields_day.split(','))

        # 连接数据库
        sqlite = SQLite(db_path)

        index = 0
        count = 300
        # 遍历沪深300
        for stock_code in list_hs_300:
            query_sql = f'SELECT date,open,high,low,close,volume,"{stock_code}" FROM "{stock_code}" ' \
                        f'WHERE date >= "{date_begin}" AND date <= "{date_end}" ORDER BY date ASC'
            list_all = sqlite.fetchall(query_sql)

            df_temp = pd.DataFrame(data=list_all, columns=fields_day.split(','))

            df = df.append(df_temp, ignore_index=True)

            index += 1
            print('load hs300 from sqlite', index, '/', count, stock_code)

            pass
        pass

        # 关闭数据库连接
        sqlite.close()

        # 更改列名
        if 'code' in df.columns:
            df = df.rename(columns={'code': 'tic'})

        # 替换空格和回车为nan
        # df.replace(to_replace=' ', value=np.nan, inplace=True)
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

        # 將nan填充为0
        df.fillna(0, inplace=True)

        df = df.sort_values(by=['date', 'tic'])

        df['open'] = df['open'].astype(np.float32)
        df['high'] = df['high'].astype(np.float32)
        df['low'] = df['low'].astype(np.float32)
        df['close'] = df['close'].astype(np.float32)
        df['volume'] = df['volume'].astype(np.int)

        return df

    @staticmethod
    def update_hs300_sqlite():
        # 获取沪深300股信息
        list_hs_300 = config.HS300_CODE_LIST

        # 连接数据库
        sqlite = SQLite(dbname=config.HS300_DB_PATH)

        stock_data = StockData()

        # 今日日期
        update_end_date = get_today_date()

        # update index
        index = 0
        count = len(list_hs_300)

        for stock_code in list_hs_300:
            # 查询是否有同名的表
            if_exists = sqlite.table_exists(stock_code)

            # 开始更新数据的日期
            update_begin_date = ''

            # 如果没有同名的表，则新建
            if if_exists is None:
                sqlite.execute_non_query(sql=f'CREATE TABLE "{stock_code}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                             f'date TEXT NOT NULL, open TEXT NOT NULL, '
                                             f'high TEXT NOT NULL, low TEXT NOT NULL, '
                                             f'close TEXT NOT NULL, volume TEXT NOT NULL);')
                # 提交
                sqlite.commit()

                # 开始更新数据的日期
                update_begin_date = '1990-01-01'
            else:
                # 如果有同名的表，获取日期最大的一条记录
                query_sql = f'SELECT date FROM "{stock_code}" ORDER BY date DESC LIMIT 1'
                max_date = sqlite.fetchone(query_sql)
                if len(max_date) > 0:
                    # 开始更新数据的日期
                    update_begin_date = max_date[0]
                pass
            pass

            # 由字符串获得日期
            date_temp = get_datetime_from_date_str(update_begin_date)
            # 获取下一个日期（不区分工作日、休息日）
            date_temp = get_next_day(datetime_date=date_temp, next_flag=+1)
            # 转成字符串格式
            update_begin_date = str(date_temp)

            # 比较日期大小，决定是否更新
            if is_greater(update_end_date, update_begin_date):
                # 下载股票数据，存入到sqlite
                raw_df = stock_data.download_raw_data(code_list=[stock_code, ],
                                                      fields=fields_day, date_start=update_begin_date,
                                                      date_end=update_end_date, frequency='d', adjustflag='3')

                # 循环 df ，写入sqlite
                for idx, row in raw_df.iterrows():
                    insert_sql = f'INSERT INTO "{stock_code}" (date, open, high, low, close, volume) ' \
                                 f'VALUES (?,?,?,?,?,?)'
                    insert_value = (row['date'], row['open'], row['high'], row['low'], row['close'], row['volume'])
                    sqlite.execute_non_query(sql=insert_sql, values=insert_value)
                pass
            pass

            index += 1
            print('update hs300 stock date', index, '/', count, stock_code)

        # 提交
        sqlite.commit()

        # 关闭数据库连接
        sqlite.close()

        # 退出baostock
        stock_data.exit()
        pass

    @staticmethod
    def insert_hs300_sqlite(list_hs_300):
        # 获取沪深300股信息

        # 连接数据库
        sqlite = SQLite(dbname=config.STOCK_DB_PATH)

        stock_data = StockData()

        # 今日日期
        update_end_date = get_today_date()

        # update index
        index = 0
        count = len(list_hs_300)

        for stock_code in list_hs_300:
            # 查询是否有同名的表
            if_exists = sqlite.table_exists(stock_code)

            # 如果没有同名的表，则新建
            if if_exists is None:
                sqlite.execute_non_query(sql=f'CREATE TABLE "{stock_code}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                             f'date TEXT NOT NULL, open TEXT NOT NULL, '
                                             f'high TEXT NOT NULL, low TEXT NOT NULL, '
                                             f'close TEXT NOT NULL, volume TEXT NOT NULL);')
                # 提交
                sqlite.commit()

                # 开始更新数据的日期
                update_begin_date = '1990-01-01'

                # 由字符串获得日期
                date_temp = get_datetime_from_date_str(update_begin_date)
                # 获取下一个日期（不区分工作日、休息日）
                date_temp = get_next_day(datetime_date=date_temp, next_flag=+1)
                # 转成字符串格式
                update_begin_date = str(date_temp)

                # 比较日期大小，决定是否更新
                if is_greater(update_end_date, update_begin_date):
                    # 下载股票数据，存入到sqlite
                    raw_df = stock_data.download_raw_data(code_list=[stock_code, ],
                                                          fields=fields_day, date_start=update_begin_date,
                                                          date_end=update_end_date, frequency='d', adjustflag='3')

                    # 循环 df ，写入sqlite
                    for idx, row in raw_df.iterrows():
                        insert_sql = f'INSERT INTO "{stock_code}" (date, open, high, low, close, volume) ' \
                                     f'VALUES (?,?,?,?,?,?)'
                        insert_value = (row['date'], row['open'], row['high'], row['low'], row['close'], row['volume'])
                        sqlite.execute_non_query(sql=insert_sql, values=insert_value)
                    pass
                pass
            else:
                # 如果有同名表，则不更新
                pass
            pass

            index += 1
            print('insert hs300 stock data', index, '/', count, stock_code)

        # 提交
        sqlite.commit()

        # 关闭数据库连接
        sqlite.close()

        # 退出baostock
        stock_data.exit()
        pass

    @staticmethod
    def save_fe_to_db(fe_df, fe_origin_table_name, if_create_or_update=False, dbname=config.STOCK_DB_PATH):
        """
        保存fe到sqlite
        :param fe_origin_table_name:
        :param dbname:
        :param fe_df: DataFrame
        :param if_create_or_update: 插入还是更新
        :return:
        """
        # 连接数据库
        sqlite = SQLite(dbname=dbname)

        index = 0
        count = len(fe_df)

        if if_create_or_update is True:
            # 直接insert
            for _, row in fe_df.iterrows():
                insert_sql = f'INSERT INTO "{fe_origin_table_name}" (date, tic, open, high, low, close, volume, ' \
                             f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma) ' \
                             f'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'

                insert_value = (
                    str(row['date']), str(row['tic']), str(row['open']), str(row['high']), str(row['low']),
                    str(row['close']), str(row['volume']), str(row['macd']), str(row['boll_ub']),
                    str(row['boll_lb']), str(row['rsi_30']), str(row['cci_30']), str(row['dx_30']),
                    str(row['close_30_sma']), str(row['close_60_sma']))

                sqlite.execute_non_query(sql=insert_sql, values=insert_value)

                index += 1
                # print('fe -> sqlite', index, '/', count)
                pass
            pass
        else:
            # 如果是更新，则先select，确认没有再insert
            # 缓存sqlite中的所有fe到dict中
            query_sql = f'SELECT date, open, high, low, close, volume, tic, ' \
                        f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, ' \
                        f'close_30_sma, close_60_sma FROM "{fe_origin_table_name}" ' \
                        f'ORDER BY date, tic ASC'

            # 查询得到的 fe 的 tuple 的 list
            list_fe_in_sqlite = sqlite.fetchall(query_sql)

            # 数据中所有的fe的缓存
            dict_fe_in_sqlite = {}

            index = 0
            count = len(list_fe_in_sqlite)

            for fe_item in list_fe_in_sqlite:
                date_temp = str(fe_item[0])
                tic_temp = str(fe_item[6])

                key = f'{tic_temp}_{date_temp}'
                dict_fe_in_sqlite[key] = ''

                index += 1
                # print('fe list -> fe dict', index, '/', count)
                pass
            pass

            index = 0
            count = len(fe_df)

            # 用dict做一个缓存，来检索
            for _, row in fe_df.iterrows():
                item_tic = row['tic']
                item_date = row['date']
                key = f'{item_tic}_{item_date}'

                if key in dict_fe_in_sqlite:
                    # 如果数据库中已经有此数据，则不操作
                    pass
                else:
                    # 数据库中没有此数据
                    insert_sql = f'INSERT INTO "{fe_origin_table_name}" (date, tic, open, high, low, close, volume, ' \
                                 f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma) ' \
                                 f'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'

                    insert_value = (
                        str(row['date']), str(row['tic']), str(row['open']), str(row['high']), str(row['low']),
                        str(row['close']), str(row['volume']), str(row['macd']), str(row['boll_ub']),
                        str(row['boll_lb']), str(row['rsi_30']), str(row['cci_30']), str(row['dx_30']),
                        str(row['close_30_sma']), str(row['close_60_sma']))

                    sqlite.execute_non_query(sql=insert_sql, values=insert_value)
                    pass
                pass
                index += 1
                # print('fe_df -> sqlite', index, '/', count)
                pass
            pass
        pass

        # 提交
        sqlite.commit()
        sqlite.close()
        pass

    @staticmethod
    def clear_and_insert_fe_to_db(fe_df, fe_origin_table_name, dbname=config.STOCK_DB_PATH):
        """
        保存fe到sqlite
        :param fe_origin_table_name:
        :param dbname:
        :param fe_df: DataFrame
        :param if_create_or_update: 插入还是更新
        :return:
        """
        # 连接数据库
        sqlite = SQLite(dbname=dbname)

        # 删除原始数据
        sqlite.execute_non_query(sql=f'DELETE FROM "{fe_origin_table_name}"')
        # 提交
        sqlite.commit()
        sqlite.close()

        sqlite = SQLite(dbname=dbname)

        index = 0
        count = len(fe_df)

        # 直接insert
        for _, row in fe_df.iterrows():
            insert_sql = f'INSERT INTO "{fe_origin_table_name}" (date, tic, open, high, low, close, volume, ' \
                         f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma) ' \
                         f'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'

            insert_value = (
                str(row['date']), str(row['tic']), str(row['open']), str(row['high']), str(row['low']),
                str(row['close']), str(row['volume']), str(row['macd']), str(row['boll_ub']),
                str(row['boll_lb']), str(row['rsi_30']), str(row['cci_30']), str(row['dx_30']),
                str(row['close_30_sma']), str(row['close_60_sma']))

            sqlite.execute_non_query(sql=insert_sql, values=insert_value)

            index += 1
            # print('fe -> sqlite', index, '/', count)
            pass
        pass

        # 提交
        sqlite.commit()
        sqlite.close()
        pass

    @staticmethod
    def download_hs300_code_list():
        """
        获取沪深300股的代码列表
        :return: list[]
        """
        bs.login()
        rs = bs.query_hs300_stocks()
        hs300_stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            row_temp = rs.get_row_data()
            hs300_stocks.append(row_temp[1])
        pass
        # result = pd.DataFrame(hs300_stocks, columns=rs.fields)
        bs.logout()
        return hs300_stocks

    @staticmethod
    def get_hs300_code_from_sqlite(table_name='hs300_list', dbname=''):
        """
        从数据库读取沪深300代码的 list
        :param table_name: 表名 hs300_list
        :return: list()
        """
        sqlite = SQLite(dbname=dbname)

        query_sql = f'SELECT hs300_list_text FROM "{table_name}" LIMIT 1'
        text_hs300 = sqlite.fetchone(query_sql)
        sqlite.close()
        list_hs300_code = text_hs300[0].split(',')

        return list_hs300_code

    @staticmethod
    def save_hs300_code_to_sqlite(list_hs300_code, table_name='hs300_list', dbname=''):
        """
        保存沪深300代码到数据库
        :param dbname:
        :param table_name: 表名称
        :param list_hs300_code: 沪深300代码 list()
        :return:
        """

        assert len(list_hs300_code) == 300

        text_hs300 = ''

        for item in list_hs300_code:
            text_hs300 += item
            text_hs300 += ','

        # 去掉最后一个逗号
        if text_hs300[-1] == ',':
            text_hs300 = text_hs300[0:-1]

        sqlite = SQLite(dbname=dbname)

        # 查询是否有同名的表
        if_exists = sqlite.table_exists(table_name)

        # 如果没有同名的表，则新建
        if if_exists is None:
            sqlite.execute_non_query(sql=f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                         f'hs300_list_text TEXT NOT NULL);')
            pass
        else:
            pass
            sqlite.execute_non_query(sql=f'DELETE FROM "{table_name}"')
        pass
        # 提交
        sqlite.commit()
        sqlite.close()

        sqlite = SQLite(dbname=dbname)

        insert_sql = f'INSERT INTO "{table_name}" (hs300_list_text) ' \
                     f'VALUES (?)'

        insert_value = (str(text_hs300),)
        sqlite.execute_non_query(sql=insert_sql, values=insert_value)

        sqlite.commit()
        sqlite.close()
        pass

    @staticmethod
    def get_fe_fillzero_from_sqlite(begin_date, end_date, table_name='fe_fillzero', date_column_name='date',
                                    code_column_name='tic', dbname=config.STOCK_DB_PATH,
                                    list_stock_code=config.SINGLE_A_STOCK_CODE):

        sqlite = SQLite(dbname=dbname)

        table_name_fillzero = table_name

        # 从数据库中再次提取到 DataFrame 中
        columns_list = fields_prep.split(',')

        list_all = []

        for item_stock_code in list_stock_code:

            query_sql = f'SELECT date, open, high, low, close, volume, tic, ' \
                        f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, ' \
                        f'close_30_sma, close_60_sma FROM "{table_name_fillzero}" ' \
                        f'WHERE tic = "{item_stock_code}" AND date >= "{begin_date}" AND date <= "{end_date}" ' \
                        f'ORDER BY date, tic ASC'

            list_single = sqlite.fetchall(query_sql)

            # 如果最后一行的 date 与 end_date 是否相同，如果不相同，则添加一条全是10的假数据
            # 注意价格，不能等于0，如果等于0，则不买/卖
            last_record_date = list_single[-1][0]
            # 假设close、volume与最后一天的close相同
            last_close_price = list_single[-1][4]
            last_volume = list_single[-1][5]

            # if last_record_date != end_date:
            while is_greater(end_date, last_record_date):
                date_temp = get_datetime_from_date_str(last_record_date)
                date_temp = get_next_work_day(datetime_date=date_temp, next_flag=+1)

                last_record_date = str(date_temp)

                # print('# 添加假数据，日期', last_record_date)

                # for item in list_stock_code:
                list_single.append((last_record_date,
                                    '10.000',
                                    '10.000',
                                    '10.000',
                                    str(last_close_price),
                                    str(last_volume),
                                    str(item_stock_code),
                                    '10.000',
                                    '10.000',
                                    '10.000',
                                    '10.000',
                                    '10.000',
                                    '10.000',
                                    '10.000',
                                    '10.000'))
                pass
            pass

            list_all += list_single
            pass
        pass
        df_result = pd.DataFrame(data=list_all, columns=columns_list)

        # 关闭数据库连接
        sqlite.close()

        df_result['open'] = df_result['open'].astype(np.float32)
        df_result['high'] = df_result['high'].astype(np.float32)
        df_result['low'] = df_result['low'].astype(np.float32)
        df_result['close'] = df_result['close'].astype(np.float32)
        df_result['volume'] = df_result['volume'].astype(np.int)

        df_result['macd'] = df_result['macd'].astype(np.float32)
        df_result['boll_ub'] = df_result['boll_ub'].astype(np.float32)
        df_result['boll_lb'] = df_result['boll_lb'].astype(np.float32)
        df_result['rsi_30'] = df_result['rsi_30'].astype(np.float32)
        df_result['cci_30'] = df_result['cci_30'].astype(np.float32)
        df_result['dx_30'] = df_result['dx_30'].astype(np.float32)
        df_result['close_30_sma'] = df_result['close_30_sma'].astype(np.float32)
        df_result['close_60_sma'] = df_result['close_60_sma'].astype(np.float32)

        # 重新索引和排序
        df_result = df_result.sort_values(by=[date_column_name, code_column_name]).reset_index(drop=True)

        return df_result

    @staticmethod
    def fill_zero_value_to_null_date(df, code_list, table_name='fe_fillzero', date_column_name='date',
                                     code_column_name='tic', dbname=config.STOCK_DB_PATH):
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

        # 唯一的日期列表
        list_date = list(column_date)

        # 存入sqlite的临时表
        # 连接数据库
        sqlite = SQLite(dbname=dbname)

        table_name_fillzero = table_name

        if_exists_table = sqlite.table_exists(table_name_fillzero)

        # 如果没有同名的表，则新建
        # date|open|high|low|close|volume|tic|macd|boll_ub|boll_lb|rsi_30|cci_30|dx_30|close_30_sma|close_60_sma
        if if_exists_table is None:
            sqlite.execute_non_query(sql=f'CREATE TABLE "{table_name_fillzero}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                         f'date TEXT NOT NULL, open TEXT NOT NULL, '
                                         f'high TEXT NOT NULL, low TEXT NOT NULL, '
                                         f'close TEXT NOT NULL, volume TEXT NOT NULL, tic TEXT NOT NULL, '
                                         f'macd TEXT NOT NULL, boll_ub TEXT NOT NULL, boll_lb TEXT NOT NULL, '
                                         f'rsi_30 TEXT NOT NULL, cci_30 TEXT NOT NULL, dx_30 TEXT NOT NULL, '
                                         f'close_30_sma TEXT NOT NULL, close_60_sma TEXT NOT NULL'
                                         f');')
        else:
            sqlite.execute_non_query(sql=f'DELETE FROM "{table_name_fillzero}"')
            pass
        pass

        # 提交
        sqlite.commit()
        sqlite.close()

        sqlite = SQLite(dbname=dbname)

        # df的全部数据，存入 dict{'code_date': 'date|open|high|low|close|volume|tic|macd|boll_ub|boll_lb|rsi_30|
        # cci_30|dx_30|close_30_sma|close_60_sma'}

        index = 0
        count = len(df)

        dict_code_with_date = {}

        for _, row in df.iterrows():
            # date_temp = str(row['date'])
            # tic_temp = str(row['tic'])

            date_temp = str(row[date_column_name])
            tic_temp = str(row[code_column_name])

            key = f'{tic_temp}_{date_temp}'
            dict_code_with_date[key] = row

            index += 1
            print(index, '/', count)
            pass
        pass

        # 循环 list_date * code_list，得到 'code_list[0]_list_date[0]' 字符串
        # 循环日期 date_list
        index = 0
        count = len(list_date) * len(code_list)

        for item_date in list_date:
            # 循环 300 支股票
            for item_tic in code_list:
                # 使用此字符串，到 dict 中寻找，是否有匹配项
                key = f'{item_tic}_{item_date}'

                # 如果找到，则 insert into 真实数据
                if key in dict_code_with_date:

                    row = dict_code_with_date[key]

                    insert_sql = f'INSERT INTO "{table_name_fillzero}" (date, tic, open, high, low, close, volume, ' \
                                 f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma) ' \
                                 f'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'

                    insert_value = (
                        str(row['date']), str(row['tic']), str(row['open']), str(row['high']), str(row['low']),
                        str(row['close']), str(row['volume']), str(row['macd']), str(row['boll_ub']),
                        str(row['boll_lb']), str(row['rsi_30']), str(row['cci_30']), str(row['dx_30']),
                        str(row['close_30_sma']), str(row['close_60_sma']))
                    pass
                else:
                    # 如果未找到,则 insert into 0 填充的数据
                    insert_sql = f'INSERT INTO "{table_name_fillzero}" (date, tic, open, high, low, close, volume, ' \
                                 f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma) ' \
                                 f'VALUES (?,?,"0","0","0","0","0","0","0","0","0","0","0","0","0")'

                    insert_value = (item_date, item_tic)
                    pass
                pass

                sqlite.execute_non_query(sql=insert_sql, values=insert_value)

                index += 1
                print(index, '/', count)
                pass
            pass
            # 提交修改
            sqlite.commit()
        pass
        sqlite.close()
        sqlite = SQLite(dbname=dbname)
        pass

        # print('从sqlite取出df')
        #
        # # 从数据库中再次提取到 DataFrame 中
        # columns_list = fields_prep.split(',')
        #
        # # df_result = pd.DataFrame(data=None, columns=columns_list)
        #
        # query_sql = f'SELECT date, open, high, low, close, volume, tic, ' \
        #             f'macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, ' \
        #             f'close_30_sma, close_60_sma FROM "{table_name_fillzero}" ' \
        #             f'ORDER BY date, tic ASC'
        #
        # list_all = sqlite.fetchall(query_sql)
        # df_result = pd.DataFrame(data=list_all, columns=columns_list)
        # # df_result = df_result.append(df_temp, ignore_index=True)
        # pass
        #
        # # 关闭数据库连接
        # sqlite.close()
        #
        # # 用0填充空白
        # # df_result.fillna(0, inplace=True)
        #
        # df_result['open'] = df_result['open'].astype(np.float32)
        # df_result['high'] = df_result['high'].astype(np.float32)
        # df_result['low'] = df_result['low'].astype(np.float32)
        # df_result['close'] = df_result['close'].astype(np.float32)
        # df_result['volume'] = df_result['volume'].astype(np.int)
        #
        # df_result['macd'] = df_result['macd'].astype(np.float32)
        # df_result['boll_ub'] = df_result['boll_ub'].astype(np.float32)
        # df_result['boll_lb'] = df_result['boll_lb'].astype(np.float32)
        # df_result['rsi_30'] = df_result['rsi_30'].astype(np.float32)
        # df_result['cci_30'] = df_result['cci_30'].astype(np.float32)
        # df_result['dx_30'] = df_result['dx_30'].astype(np.float32)
        # df_result['close_30_sma'] = df_result['close_30_sma'].astype(np.float32)
        # df_result['close_60_sma'] = df_result['close_60_sma'].astype(np.float32)
        #
        # # print('重新索引和排序')
        # # 重新索引和排序
        # df_result = df_result.sort_values(by=[date_column_name, code_column_name]).reset_index(drop=True)
        #
        # return df_result

    @staticmethod
    def update_batch_stock_sqlite(list_stock_code, dbname=config.STOCK_DB_PATH, adjustflag='2'):
        """
        更新 批量 股票数据，直到今天
        :param list_stock_code: 股票代码List
        :param dbname: 数据库文件地址
        """
        # 连接数据库
        sqlite = SQLite(dbname=dbname)

        # 初始化baostock
        stock_data = StockData()

        # 今日日期
        update_end_date = get_today_date()

        # update index
        index = 0
        count = len(list_stock_code)

        for stock_code in list_stock_code:
            # 查询是否有同名的表
            if_exists = sqlite.table_exists(stock_code)

            # 开始更新数据的日期
            update_begin_date = ''

            # 如果没有同名的表，则新建
            if if_exists is None:
                sqlite.execute_non_query(sql=f'CREATE TABLE "{stock_code}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                             f'date TEXT NOT NULL, open TEXT NOT NULL, '
                                             f'high TEXT NOT NULL, low TEXT NOT NULL, '
                                             f'close TEXT NOT NULL, volume TEXT NOT NULL);')
                # 提交
                sqlite.commit()

                # 开始更新数据的日期
                update_begin_date = '1990-01-01'
            else:
                # 如果有同名的表，获取日期最大的一条记录
                query_sql = f'SELECT date FROM "{stock_code}" ORDER BY date DESC LIMIT 1'
                max_date = sqlite.fetchone(query_sql)
                if max_date is not None:
                    if len(max_date) > 0:
                        # 开始更新数据的日期
                        update_begin_date = max_date[0]
                    pass
                pass
            pass

            # 由字符串获得日期
            date_temp = get_datetime_from_date_str(update_begin_date)
            # 获取下一个日期（不区分工作日、休息日）
            date_temp = get_next_day(datetime_date=date_temp, next_flag=+1)
            # 转成字符串格式
            update_begin_date = str(date_temp)

            # 比较日期大小，决定是否更新
            if is_greater(update_end_date, update_begin_date):
                # 下载股票数据，存入到sqlite
                raw_df = stock_data.download_raw_data(code_list=[stock_code, ],
                                                      fields=fields_day, date_start=update_begin_date,
                                                      date_end=update_end_date, frequency='d', adjustflag=adjustflag)

                # 如果为空，则下一个循环
                if raw_df is None or raw_df.empty is True:
                    pass
                else:
                    # 循环 df ，写入sqlite
                    for idx, row in raw_df.iterrows():
                        insert_sql = f'INSERT INTO "{stock_code}" (date, open, high, low, close, volume) ' \
                                     f'VALUES (?,?,?,?,?,?)'
                        insert_value = (row['date'], row['open'], row['high'], row['low'], row['close'], row['volume'])
                        sqlite.execute_non_query(sql=insert_sql, values=insert_value)
                    pass
                pass
            pass

            index += 1
            print('update batch stock data', index, '/', count, stock_code)
        pass

        # 提交
        sqlite.commit()

        # 关闭数据库连接
        sqlite.close()

        # 退出baostock
        stock_data.exit()
        pass

    @staticmethod
    def load_stock_raw_data_from_sqlite(list_batch_code, date_begin='', date_end='', db_path=config.STOCK_DB_PATH):
        """
        从sqlite数据库加载批量股票数据
        :param list_batch_code: 股票代码List
        :param date_begin: 开始日期
        :param date_end: 结束日期
        :param db_path: 数据库地址
        :return: DataFrame
        """
        df = pd.DataFrame(data=None, columns=fields_day.split(','))

        # 连接数据库
        sqlite = SQLite(db_path)

        index = 0
        count = len(list_batch_code)

        # 遍历股票代码List
        for stock_code in list_batch_code:
            query_sql = f'SELECT date,open,high,low,close,volume,"{stock_code}" FROM "{stock_code}" ' \
                        f'WHERE date >= "{date_begin}" AND date <= "{date_end}" ORDER BY date ASC'
            list_all = sqlite.fetchall(query_sql)
            df_temp = pd.DataFrame(data=list_all, columns=fields_day.split(','))
            df = df.append(df_temp, ignore_index=True)

            index += 1
            # print('load batch stock from sqlite', index, '/', count, stock_code)
            pass
        pass

        # 关闭数据库连接
        sqlite.close()

        # 更改列名
        if 'code' in df.columns:
            df = df.rename(columns={'code': 'tic'})

        # 替换空格和回车为nan
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

        # 將nan填充为0
        df.fillna(0, inplace=True)

        df = df.sort_values(by=['date', 'tic'])

        df['open'] = df['open'].astype(np.float32)
        df['high'] = df['high'].astype(np.float32)
        df['low'] = df['low'].astype(np.float32)
        df['close'] = df['close'].astype(np.float32)
        df['volume'] = df['volume'].astype(np.int)

        return df

    @staticmethod
    def append_rows_to_raw_df(raw_df, rows, columns=None):
        # 添加rows到df
        if columns is None:
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        pass

        df_rows = pd.DataFrame(data=rows, columns=columns)
        df_rows['open'] = df_rows['open'].astype(np.float32)
        df_rows['high'] = df_rows['high'].astype(np.float32)
        df_rows['low'] = df_rows['low'].astype(np.float32)
        df_rows['close'] = df_rows['close'].astype(np.float32)
        df_rows['volume'] = df_rows['volume'].astype(np.int)

        # date,open,high,low,close,volume,tic
        raw_df = raw_df.append(df_rows, ignore_index=True)
        return raw_df
        pass

    @staticmethod
    def push_predict_result_to_psql(psql, agent, vali_period_value, pred_period_name, tic, date, action, hold, day):

        # list_buy_or_sell_output
        # tic, date, agent, vali_period_value, pred_period_name, sell_buy, hold
        # tic, date, sell/buy, hold, 第x天

        sql_cmd = f'INSERT INTO "public"."{tic}" ("date", "agent", "vali_period_value", ' \
                  f'"pred_period_name", "action", "hold", "day") VALUES (%s,%s,%s,%s,%s,%s,%s)'

        sql_values = (str(date), agent, vali_period_value, pred_period_name, str(action), str(hold), str(day))

        psql.execute_non_query(sql=sql_cmd, values=sql_values)
        psql.commit()
        pass

    pass

    @staticmethod
    def create_predict_result_table_psql(tic=''):
        psql = Psqldb(database=config.PSQL_DATABASE, user=config.PSQL_USER,
                      password=config.PSQL_PASSWORD, host=config.PSQL_HOST, port=config.PSQL_PORT)

        sql_cmd = f'CREATE TABLE "public"."{tic}" ' \
                  f'("id" serial8, "date" date NOT NULL, "agent" text NOT NULL, ' \
                  f'"vali_period_value" int4 NOT NULL, "pred_period_name" text NOT NULL, ' \
                  f'"action" decimal NOT NULL, "hold" decimal NOT NULL, "day" int4 NOT NULL, ' \
                  f'PRIMARY KEY ("id"));'

        # sql_value = ''
        psql.execute_non_query(sql=sql_cmd)
        psql.commit()
        psql.close()
        pass

    @staticmethod
    def get_today_stock_data_from_sina_api(tic_code='', date=''):
        url = f'http://hq.sinajs.cn/list={tic_code}'
        res = requests.get(url)
        list1 = res.text.split('"')

        if len(list1) == 3:
            text1 = list1[1]
            list2 = text1.split(',')

            if len(list2) == 34:
                if list2[30] == date:
                    open1 = list2[1]
                    high1 = list2[4]
                    low1 = list2[5]
                    close1 = list2[6]
                    volume1 = list2[8]
                    return open1, high1, low1, close1, volume1
                pass
            pass
        pass

        return None, None, None, None, None
    pass


if __name__ == '__main__':
    pass
