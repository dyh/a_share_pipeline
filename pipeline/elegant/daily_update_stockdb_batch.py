import sys

if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../ElegantRL_master')

from pipeline.elegant.env_train_batch import FeatureEngineer
from pipeline.utils.datetime import get_today_date, get_next_day, get_datetime_from_date_str

from pipeline.stock_data import StockData

from pipeline.elegant import config
from ElegantRL_master.elegantrl.run import *

if __name__ == '__main__':
    # 开始训练的日期，在程序启动之后，不要改变。
    # 在 daily_update_stockdb_batch 里，修改 config.BATCH_A_STOCK_CODE list ，和 训练-预测-结束 日期。
    # config.BATCH_A_STOCK_CODE = ['sh.600036', 'sh.600295', ]
    config.BATCH_A_STOCK_CODE = ['sh.600036', ]

    config.START_DATE = "2002-05-01"
    # config.START_EVAL_DATE = "2021-03-12"
    config.END_DATE = "2021-05-13"

    # 下载、更新 股票数据
    StockData.update_batch_stock_sqlite(config.BATCH_A_STOCK_CODE)

    # 缓存 raw 数据 为 df 。
    raw_df = StockData.load_batch_stock_from_sqlite(list_batch_code=config.BATCH_A_STOCK_CODE,
                                                    date_begin=config.START_DATE, date_end=config.END_DATE,
                                                    db_path=config.BATCH_DB_PATH)

    # raw_df -> fe
    fe_origin_table_name = "fe_origin"

    # 创建fe表
    StockData.create_fe_table(db_path=config.BATCH_DB_PATH, table_name=fe_origin_table_name)

    fe = FeatureEngineer(use_turbulence=False,
                         user_defined_feature=False,
                         use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST, )

    fe_df = fe.preprocess_data(raw_df)
    # 保存pickle
    # fe_df.to_pickle('raw_to_fe.df')
    # fe_df = pd.read_pickle('raw_to_fe.df')

    # 将 fe_df 存入数据库
    # 增量fe
    StockData.save_fe_to_db(fe_df, fe_origin_table_name=fe_origin_table_name)

    # fe -> fillzero
    # 增量fillzero
    fillzero_df = StockData.fill_zero_value_to_null_date(df=fe_df, code_list=config.BATCH_A_STOCK_CODE,
                                                         table_name='fe_fillzero', date_column_name='date',
                                                         code_column_name='tic')
    # 保存pickle
    # 将 fillzero_df 存入数据库

    print('done!')

    pass
