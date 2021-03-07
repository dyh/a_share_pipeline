import baostock as bs
import pandas as pd
import os
import config as config
import numpy as np


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class StockData(object):
    def __init__(self, output_dir, date_start, date_end):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        # ,date,open,high,low,close,volume,tic,day
        # self.fields = "date,open,high,low,close,volume,code,amount," \
        #               "adjustflag,turn,tradestatus,pctChg,peTTM," \
        #               "pbMRQ,psTTM,pcfNcfTTM,isST"

        self.fields = "date,open,high,low,close,volume,code,amount,turn,tradestatus,pctChg,peTTM, pbMRQ,psTTM,pcfNcfTTM"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def download(self, code):
        df = bs.query_history_k_data_plus(code, self.fields,
                                          start_date=self.date_start,
                                          end_date=self.date_end).get_data()

        # 删除停牌日期的行，即 tradestatus=0 的数据，只保留 tradestatus=1 的数据
        df = df[df['tradestatus'] == '1']

        # 删除 tradestatus 列、adjustflag 列、isST 列
        df.drop(['tradestatus'], axis=1, inplace=True)
        # df.drop(['tradestatus', 'adjustflag', 'isST'], axis=1, inplace=True)

        # 更改列名
        df = df.rename(columns={'code': 'tic'})

        # 替换空格和回车为nan
        # df.replace(to_replace=' ', value=np.nan, inplace=True)
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

        # 將nan填充为0
        df.fillna(0, inplace=True)
        # 统计nan值
        print(df.isnull().sum())

        csv_file_path = f'{self.output_dir}/{code}.csv'
        df.to_csv(csv_file_path, index=False)

        # 下载所有股票数据
        # stock_df = self.get_codes_by_date(self.date_end)
        # for index, row in stock_df.iterrows():
        #     print(f'processing {row["code"]} {row["code_name"]}')

        self.exit()

        return csv_file_path
