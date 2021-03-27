import sys

if 'Informer2020_main' not in sys.path:
    sys.path.append('../../Informer2020_main')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

from pipeline.stock_data import StockData
import pipeline.utils.datetime

if __name__ == '__main__':
    # 下载数据
    download_data = True

    # 使用技术指标
    use_technical_indicator = False

    output_dir = './temp_dataset'
    output_file_name = 'ETTm1.csv'

    # 下载A股数据
    # 处理数据，成为OT的格式
    stock_code = 'sh.600036'
    date_start = '2002-05-01'
    # date_end = '2021-03-19'
    date_end = pipeline.utils.datetime.get_today_date()

    if download_data is True:
        stock = StockData(output_dir=output_dir, date_start=date_start, date_end=date_end)
        stock.get_informer_data(stock_code=stock_code, fields=stock.fields_minutes,
                                frequency='30', adjustflag='3', output_file_name=output_file_name,
                                use_technical_indicator=use_technical_indicator)

    print('done!')
    pass
