import pymysql
import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()

#股票编码
Str_Share_Code = 'sh.600000'

rs = bs.query_history_k_data_plus(Str_Share_Code,
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
    start_date='', end_date='',
    frequency="d", adjustflag="3")

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())


#连接数据库
conn = pymysql.connect(host="localhost", user="root", password="root_hao_1225", database="rl",charset="utf8")
cur = conn.cursor()

#查询库里有没有此表
str_SQl = "SHOW TABLES LIKE '"+Str_Share_Code+"';"
#执行创建表
cur.execute(str_SQl)
#搜索表的结果
rs=cur.fetchall()

#如果如果数据里，有此表则不执行
if 0 >= len(rs):
    # 创建表
    SQl_Create_Table="CREATE TABLE IF NOT EXISTS `"+Str_Share_Code+"` (`id` INT UNSIGNED AUTO_INCREMENT,`date` DATE, `code` varchar(30) not null,`open` varchar(30) not null,`high`varchar(30) not null ,`low` varchar(30) not null,`close` varchar(30) not null,`preclose` varchar(30) not null ,`volume` varchar(30) not null,`amount` varchar(30) not null,`adjustflag` varchar(30) not null,`turn` varchar(30) not null,`tradestatus` varchar(30) not null,`pctChg` varchar(30) not null,`peTTM` varchar(30) not null,`pbMRQ` varchar(30) not null,`psTTM` varchar(30) not null,`pcfNcfTTM` varchar(30) not null,`isST` varchar(30) not null,PRIMARY KEY (`id`));"
    #执行SQL语句——创建表
    cur.execute(SQl_Create_Table)
    #将股票信息循环写入数据库
    i = 0
    while len(data_list) > i:
        #print(i)
        # 添加信息
        Sql_Add="INSERT INTO `"+Str_Share_Code+"` (`date`,`code`,`open`,`high`,`low`,`close`,`preclose`,`volume`,`amount`,`adjustflag`,`turn`,`tradestatus`,`pctChg`,`peTTM`,`pbMRQ`,`psTTM`,`pcfNcfTTM`,`isST`) VALUES ('"+data_list[i][0]+"','"+data_list[i][1]+"','"+data_list[i][2]+"','"+data_list[i][3]+"','"+data_list[i][4]+"','"+data_list[i][5]+"','"+data_list[i][6]+"','"+data_list[i][7]+"','"+data_list[i][8]+"','"+data_list[i][9]+"','"+data_list[i][10]+"','"+data_list[i][11]+"','"+data_list[i][12]+"','"+data_list[i][13]+"','"+data_list[i][14]+"','"+data_list[i][15]+"','"+data_list[i][16]+"','"+data_list[i][17]+"');"
        # 执行SQL语句
        cur.execute(Sql_Add)
        i = i + 1
    # 执行SQL语句(不使用该代码，可能会使信息无法写入)
    conn.commit()
    #释放资源
    cur.close()
    conn.close()
    #完成
    print("done!")
else:
    print("此表已存在")