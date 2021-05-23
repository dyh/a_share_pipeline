import sys

if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

if 'ElegantRL_master' not in sys.path:
    sys.path.append('../../ElegantRL_master')

from pipeline.elegant import config

from pipeline.elegant.env_train_batch import FeatureEngineer

from pipeline.stock_data import StockData


if __name__ == '__main__':

    # --------------------------------------------------------------
    # 要预测的股票
    config.BATCH_A_STOCK_CODE = ['sh.600036', ]

    # 默认初始数据日期，对预测无影响
    config.START_DATE = "2002-05-01"

    # 要预测的日期
    config.END_DATE = "2021-05-24"

    # --------------------------------------------------------------
    # config.BATCH_A_STOCK_CODE = ['sh.600036', 'sh.600295', ]
    # config.START_EVAL_DATE = "2021-03-12"

    # 下载、更新 股票数据
    StockData.update_batch_stock_sqlite(list_stock_code=config.BATCH_A_STOCK_CODE, dbname=config.STOCK_DB_PATH)

    # 缓存 raw 数据 为 df 。
    raw_df = StockData.load_stock_raw_data_from_sqlite(list_batch_code=config.BATCH_A_STOCK_CODE,
                                                       date_begin=config.START_DATE, date_end=config.END_DATE,
                                                       db_path=config.STOCK_DB_PATH)

    # raw_df -> fe
    fe_origin_table_name = "fe_origin"

    # 创建fe表
    StockData.create_fe_table(db_path=config.STOCK_DB_PATH, table_name=fe_origin_table_name)

    fe = FeatureEngineer(use_turbulence=False,
                         user_defined_feature=False,
                         use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST, )

    fe_df = fe.preprocess_data(raw_df)

    # 将 fe_df 存入数据库
    # 增量fe
    # StockData.save_fe_to_db(fe_df, fe_origin_table_name=fe_origin_table_name, dbname=config.STOCK_DB_PATH)
    StockData.clear_and_insert_fe_to_db(fe_df, fe_origin_table_name=fe_origin_table_name)

    # # 单支预测，不用fill_zero
    # # fe -> fillzero
    # fillzero_df = StockData.fill_zero_value_to_null_date(df=fe_df, code_list=config.BATCH_A_STOCK_CODE,
    #                                                      table_name='fe_fillzero', date_column_name='date',
    #                                                      code_column_name='tic', dbname=config.STOCK_DB_PATH)

    print('done!')

    pass
