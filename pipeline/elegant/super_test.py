from pipeline.elegant import config
from pipeline.sqlite import SQLite
from pipeline.stock_data import StockData

if __name__ == '__main__':

    # # 下载hs300到数据库
    # # str1 = 'sh.600000,sh.600004,sh.600009,sh.600010,sh.600011,sh.600015,sh.600016,sh.600018,sh.600019,sh.600025,sh.600027,sh.600028,sh.600029,sh.600030,sh.600031,sh.600036,sh.600048,sh.600050,sh.600061,sh.600066,sh.600068,sh.600085,sh.600104,sh.600109,sh.600111,sh.600115,sh.600118,sh.600150,sh.600161,sh.600176,sh.600177,sh.600183,sh.600196,sh.600208,sh.600233,sh.600271,sh.600276,sh.600297,sh.600299,sh.600309,sh.600332,sh.600340,sh.600346,sh.600352,sh.600362,sh.600369,sh.600383,sh.600390,sh.600406,sh.600436,sh.600438,sh.600482,sh.600487,sh.600489,sh.600498,sh.600519,sh.600522,sh.600547,sh.600570,sh.600584,sh.600585,sh.600588,sh.600600,sh.600606,sh.600637,sh.600655,sh.600660,sh.600690,sh.600703,sh.600705,sh.600741,sh.600745,sh.600760,sh.600763,sh.600795,sh.600809,sh.600837,sh.600845,sh.600848,sh.600872,sh.600886,sh.600887,sh.600893,sh.600900,sh.600918,sh.600919,sh.600926,sh.600958,sh.600989,sh.600998,sh.600999,sh.601006,sh.601009,sh.601012,sh.601021,sh.601066,sh.601077,sh.601088,sh.601100,sh.601108,sh.601111,sh.601117,sh.601138,sh.601155,sh.601162,sh.601166,sh.601169,sh.601186,sh.601198,sh.601211,sh.601216,sh.601225,sh.601229,sh.601231,sh.601236,sh.601238,sh.601288,sh.601318,sh.601319,sh.601328,sh.601336,sh.601360,sh.601377,sh.601390,sh.601398,sh.601555,sh.601577,sh.601600,sh.601601,sh.601607,sh.601618,sh.601628,sh.601633,sh.601658,sh.601668,sh.601669,sh.601688,sh.601696,sh.601698,sh.601727,sh.601766,sh.601788,sh.601800,sh.601808,sh.601816,sh.601818,sh.601838,sh.601857,sh.601872,sh.601877,sh.601878,sh.601881,sh.601888,sh.601899,sh.601901,sh.601916,sh.601919,sh.601933,sh.601939,sh.601985,sh.601988,sh.601989,sh.601990,sh.601998,sh.603019,sh.603087,sh.603156,sh.603160,sh.603195,sh.603259,sh.603288,sh.603369,sh.603392,sh.603501,sh.603658,sh.603799,sh.603833,sh.603899,sh.603986,sh.603993,sh.688008,sh.688009,sh.688012,sh.688036,sz.000001,sz.000002,sz.000063,sz.000066,sz.000069,sz.000100,sz.000157,sz.000166,sz.000333,sz.000338,sz.000425,sz.000538,sz.000568,sz.000596,sz.000625,sz.000627,sz.000651,sz.000656,sz.000661,sz.000671,sz.000703,sz.000708,sz.000723,sz.000725,sz.000728,sz.000768,sz.000776,sz.000783,sz.000786,sz.000858,sz.000860,sz.000876,sz.000895,sz.000938,sz.000961,sz.000963,sz.000977,sz.001979,sz.002001,sz.002007,sz.002008,sz.002024,sz.002027,sz.002032,sz.002044,sz.002049,sz.002050,sz.002120,sz.002129,sz.002142,sz.002146,sz.002153,sz.002157,sz.002179,sz.002202,sz.002230,sz.002236,sz.002241,sz.002252,sz.002271,sz.002304,sz.002311,sz.002352,sz.002371,sz.002384,sz.002410,sz.002414,sz.002415,sz.002422,sz.002456,sz.002460,sz.002463,sz.002475,sz.002493,sz.002508,sz.002555,sz.002558,sz.002594,sz.002600,sz.002601,sz.002602,sz.002607,sz.002624,sz.002673,sz.002714,sz.002736,sz.002739,sz.002773,sz.002812,sz.002821,sz.002841,sz.002916,sz.002938,sz.002939,sz.002945,sz.002958,sz.003816,sz.300003,sz.300014,sz.300015,sz.300033,sz.300059,sz.300122,sz.300124,sz.300136,sz.300142,sz.300144,sz.300347,sz.300408,sz.300413,sz.300433,sz.300498,sz.300529,sz.300601,sz.300628,sz.300676'
    #
    # 从baostock下载到sqlite
    hs300_code_list = StockData.download_hs300_code_list()
    #
    # StockData.save_hs300_code_to_sqlite(list_hs300_code=hs300_code_list, table_name='hs300_list',
    #                                     dbname=config.STOCK_DB_PATH)

    config.HS300_CODE_LIST = hs300_code_list
    #
    # print(config.HS300_CODE_LIST)
    #
    StockData.insert_hs300_sqlite(config.HS300_CODE_LIST)
    #
    # print('done!')

    # 查找hs300，2002年5月1日之前的hs300,有哪些。
    hs300_code_list = StockData.get_hs300_code_from_sqlite(table_name='hs300_list', dbname=config.STOCK_DB_PATH)

    sqlite = SQLite(dbname=config.STOCK_DB_PATH)

    index = 0
    for stock_code in hs300_code_list:
        query_sql = f'SELECT date FROM "{stock_code}" WHERE date <= "2002-05-01" LIMIT 1'
        text_date = sqlite.fetchone(query_sql)
        if text_date is not None:
            index += 1
            print(text_date, stock_code)
        pass

    print('count', index)
    sqlite.close()

    pass
