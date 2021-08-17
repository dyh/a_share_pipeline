import baostock as bs
import pandas as pd

from stock_data import StockData

if __name__ == '__main__':

    list1 = StockData.get_batch_a_share_code_list_string(date_filter='2004-05-01')

    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取行业分类数据
    rs = bs.query_stock_industry()
    # rs = bs.query_stock_basic(code_name="浦发银行")
    print('query_stock_industry error_code:' + rs.error_code)
    print('query_stock_industry respond  error_msg:' + rs.error_msg)

    # 打印结果集
    industry_list = []
    # 查询 季频盈利能力
    profit_list = []

    fields1 = ['code', 'code_name', 'industry', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin',
               'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare']

    index1 = 0
    count1 = len(list1)

    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        industry_item = rs.get_row_data()

        updateDate, code, code_name, industry, industryClassification = industry_item

        if code in list1:

            rs_profit = bs.query_profit_data(code=code, year=2021, quarter=1)

            while (rs_profit.error_code == '0') & rs_profit.next():
                code, pubDate, statDate, roeAvg, npMargin, gpMargin, netProfit, epsTTM, MBRevenue, totalShare, liqaShare = rs_profit.get_row_data()

                # profit_list.append(rs_profit.get_row_data())
                profit_list.append(
                    [code, code_name, industry, pubDate, statDate, roeAvg, npMargin, gpMargin, netProfit, epsTTM,
                     MBRevenue, totalShare, liqaShare])

            pass
        pass

        index1 += 1
        print(index1, '/', count1)

    pass

    result_profit = pd.DataFrame(profit_list, columns=fields1)

    # result = pd.DataFrame(industry_list, columns=rs.fields)

    # 结果集输出到csv文件
    result_profit.to_csv("./stock_industry_profit.csv", index=False)

    print(result_profit)

    # 登出系统
    bs.logout()
    pass
