import sys

if 'pipeline' not in sys.path:
    sys.path.append('../../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../../ElegantRL_master')

from pipeline.elegant.hs300.env_train_hs300 import FeatureEngineer
from pipeline.utils.date_time import get_today_date, get_next_day, get_datetime_from_date_str

from pipeline.stock_data import StockData

from pipeline.elegant import config
from elegantrl.run import *

if __name__ == '__main__':

    # 是否第一次运行
    # 如果是第一次运行，processed_df和fillzero，先清空表内所有内容，再insert into，即不查询是否有数据
    if_create_or_update = False

    # # 从baostock下载到sqlite
    # hs300_code_list = StockData.download_hs300_code_list()
    # # 保存hs300股票代码字符串到数据库
    # StockData.save_hs300_code_to_sqlite(hs300_code_list)

    config.HS300_CODE_LIST = StockData.get_hs300_code_from_sqlite()

    today_date = get_today_date()
    # 从sqlite读取到df
    if if_create_or_update is True:
        # 足够长的日期，读取所有数据
        update_begin_date = '1990-01-01'
        update_date_end = today_date
    else:
        # 更新2天的数据
        update_begin_date = get_next_day(get_datetime_from_date_str(today_date), next_flag=-2)
        update_date_end = today_date
        pass
    pass

    # raw_df = StockData.load_hs300_from_sqlite(db_path=config.HS300_DB_PATH, list_hs_300=config.HS300_CODE_LIST,
    #                                           date_begin=update_begin_date, date_end=update_date_end)

    # hs300 -> fe
    fe_origin_table_name = "fe_origin"

    if if_create_or_update is True:
        # 如果是第一次运行，则创建fe表
        # StockData.create_fe_table(db_path=config.HS300_DB_PATH, table_name=fe_raw_table_name)
        pass
    else:
        pass
    pass

    fe = FeatureEngineer(use_turbulence=False,
                         user_defined_feature=False,
                         use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST, )

    # fe_df = fe.preprocess_data(raw_df)
    # 保存pickle
    # fe_df.to_pickle('raw_to_fe.df')

    # fe_df = pd.read_pickle('raw_to_fe.df')

    # 将 fe_df 存入数据库
    # StockData.save_fe_to_db(fe_df, if_create_or_update, fe_origin_table_name)

    # fe -> fillzero
    fillzero_df = StockData.fill_zero_value_to_null_date(df=fe_df, code_list=config.HS300_CODE_LIST,
                                                         date_column_name='date', code_column_name='tic')

    # 保存pickle
    # 将 fillzero_df 存入数据库

    print('done!')

    pass
