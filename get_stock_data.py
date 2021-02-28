import baostock as bs
import pandas as pd
import os
import config_a_share as config
import numpy as np


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self, output_dir, date_start, date_end):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        # ,date,open,high,low,close,volume,tic,day
        self.fields = "date,open,high,low,close,volume,code,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self, code):
        df = bs.query_history_k_data_plus(code, self.fields,
                                          start_date=self.date_start,
                                          end_date=self.date_end).get_data()

        # 更改列名
        df = df.rename(columns={'code': 'tic'})

        # 替换空格和回车为nan
        # df.replace(to_replace=' ', value=np.nan, inplace=True)
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

        # 填充nan为0
        df.fillna(0, inplace=True)
        # 统计nan值
        # print(df.isnull().sum())

        # 删除停牌日期的行，即 tradestatus=0 的数据
        df = df[df['tradestatus'] == '1']

        df.to_csv(f'{self.output_dir}/{code}.csv', index=False)

        # stock_df = self.get_codes_by_date(self.date_end)
        # for index, row in stock_df.iterrows():
        #     print(f'processing {row["code"]} {row["code_name"]}')

        self.exit()


if __name__ == '__main__':
    stock_code = 'sh.600036'

    test_csv_file_path = "./" + config.DATA_SAVE_DIR + '/' + stock_code + '.csv'

    # 获取股票的日K线数据
    downloader = Downloader("./" + config.DATA_SAVE_DIR, date_start=config.START_DATE, date_end=config.END_DATE)
    downloader.run(stock_code)

    print('done!')
