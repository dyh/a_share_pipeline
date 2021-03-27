import sys

sys.path.append('../FinRL_Library_master')

import baostock as bs
import pandas as pd
import os
from FinRL_Library_master.finrl.config import config
import numpy as np

from FinRL_Library_master.finrl.preprocessing.preprocessors import FeatureEngineer


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

        self.fields_day = "date,open,high,low,close,volume,code,amount,turn," \
                          "tradestatus,pctChg,peTTM, pbMRQ,psTTM,pcfNcfTTM"

        self.fields_minutes = "time,open,high,low,volume,amount,code,close"

        # 15分钟K线的时间点
        self.list_time_point_15minutes = ['09:45:00',
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

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

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
