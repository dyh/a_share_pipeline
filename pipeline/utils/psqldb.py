# coding=utf-8
import time

import logging
import psycopg2


# 判断是否数据库超时或者数据库连接关闭
def is_timeout_or_closed_error(str_exception):
    if "Connection timed out" in str_exception or "connection already closed" in str_exception:
        return True
    else:
        return False


class Psqldb:
    def __init__(self, database, user, password, host, port):
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        # 调用连接函数
        self.re_connect()

    # 再次连接数据库
    def re_connect(self):
        # 如果连接数据库超时，则循环再次连接
        while True:
            try:
                print(__name__, "re_connect")

                self.conn = psycopg2.connect(database=self.database, user=self.user,
                                             password=self.password, host=self.host, port=self.port)
                # 如果连接正常，则退出循环
                break
            except psycopg2.OperationalError as e:
                logging.exception(e)
            except Exception as e:
                # 其他错误，记录，退出
                logging.exception(e)
                break

            # sleep 5秒后重新连接
            print(__name__, "re-connecting PostgreSQL in 5 seconds")
            time.sleep(5)

    # 提交数据
    def commit(self):
        try:
            self.conn.commit()
        except psycopg2.OperationalError as e:
            logging.exception(e)
        except Exception as e:
            logging.exception(e)

    # 关闭数据库
    def close(self):
        try:
            print(__name__, "close")

            self.conn.close()
        except psycopg2.OperationalError as e:
            logging.exception(e)
        except Exception as e:
            logging.exception(e)

    # 执行sql但不查询
    def execute_non_query(self, sql, values=()):
        result = False
        while True:
            try:
                # console(__name__, "execute_non_query")

                cursor = self.conn.cursor()
                cursor.execute(sql, values)
                # self.commit()
                result = True
                break
            except psycopg2.OperationalError as e:
                logging.exception(e)
                str_exception = str(e)
                if is_timeout_or_closed_error(str_exception):
                    # 重新连接数据库
                    self.re_connect()
                else:
                    break
            except Exception as e:
                # 其他错误，记录，退出
                logging.exception(e)
                str_exception = str(e)
                if is_timeout_or_closed_error(str_exception):
                    # console(__name__, "re-connecting PostgreSQL immediately")
                    # 重新连接数据库
                    self.re_connect()
                else:
                    break
            # sleep 5秒后重新连接
            print(__name__, "re-connecting PostgreSQL in 5 seconds")
            time.sleep(5)

        return result

    # 查询，返回所有结果
    def fetchall(self, sql, values=()):
        results = None
        while True:
            try:
                # console(__name__, "fetchall")

                cursor = self.conn.cursor()
                cursor.execute(sql, values)
                results = cursor.fetchall()
                break
            except psycopg2.OperationalError as e:
                logging.exception(e)
                str_exception = str(e)
                if is_timeout_or_closed_error(str_exception):
                    # 重新连接数据库
                    self.re_connect()
                else:
                    break
            except Exception as e:
                # 其他错误，记录忽略
                logging.exception(e)
                str_exception = str(e)
                if is_timeout_or_closed_error(str_exception):
                    # 重新连接数据库
                    self.re_connect()
                else:
                    break
        return results

    # 查询，返回一条结果
    def fetchone(self, sql, values=()):
        result = None
        while True:
            try:
                # console(__name__, "fetchone")

                cursor = self.conn.cursor()
                cursor.execute(sql, values)
                result = cursor.fetchone()
                break
            except psycopg2.OperationalError as e:
                logging.exception(e)
                str_exception = str(e)
                if is_timeout_or_closed_error(str_exception):
                    # 重新连接数据库
                    self.re_connect()
                else:
                    break
            except Exception as e:
                # 其他错误，记录，退出
                logging.exception(e)
                str_exception = str(e)
                if is_timeout_or_closed_error(str_exception):
                    # 重新连接数据库
                    self.re_connect()
                else:
                    break
        return result

    # # 查询，返回一条结果
    # def table_exists(self, table_name):
    #     try:
    #         cursor = self.conn.cursor()
    #         cursor.execute("select name from sqlite_master where type = 'table' and name = '{}'".format(table_name))
    #         value = cursor.fetchone()
    #         return value
    #     except psycopg2.OperationalError as e:
    #         logging.exception(e)
    #         pass
    #     except Exception as e:
    #         logging.exception(e)
    #     finally:
    #         pass
    #
    # # 删除一个表
    # def drop_table(self, table_name):
    #     value = False
    #     try:
    #         cursor = self.conn.cursor()
    #         cursor.execute("DROP TABLE IF EXISTS '{}'".format(table_name))
    #         value = True
    #     except psycopg2.OperationalError as e:
    #         logging.exception(e)
    #         pass
    #     except Exception as e:
    #         logging.exception(e)
    #     finally:
    #         return value
