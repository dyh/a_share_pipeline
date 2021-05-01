# coding = utf-8

import logging
import sqlite3


class SQLite:
    def __init__(self, dbname):
        self.name = dbname
        self.conn = sqlite3.connect(self.name, check_same_thread=False)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    # 执行sql但不查询
    def execute_non_query(self, sql, values=()):
        value = False
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, values)
            # self.commit()
            value = True
        except sqlite3.OperationalError as e:
            logging.exception(e)
            pass
        except Exception as e:
            logging.exception(e)
        finally:
            return value
            # self.close()

    # 查询，返回所有结果
    def fetchall(self, sql, values=()):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, values)
            values = cursor.fetchall()
            return values
        except sqlite3.OperationalError as e:
            logging.exception(e)
            pass
        except Exception as e:
            logging.exception(e)
        finally:
            pass

    # 查询，返回一条结果
    def fetchone(self, sql, values=()):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, values)
            value = cursor.fetchone()
            return value
        except sqlite3.OperationalError as e:
            logging.exception(e)
            pass
        except Exception as e:
            logging.exception(e)
        finally:
            pass

    # 查询，返回一条结果
    def table_exists(self, table_name):
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"select name from sqlite_master where type = 'table' and name = '{table_name}'")
            value = cursor.fetchone()
            return value
        except sqlite3.OperationalError as e:
            logging.exception(e)
            pass
        except Exception as e:
            logging.exception(e)
        finally:
            pass

    # 删除一个表
    def drop_table(self, table_name):
        value = False
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS '{table_name}'")
            value = True
        except sqlite3.OperationalError as e:
            logging.exception(e)
            pass
        except Exception as e:
            logging.exception(e)
        finally:
            return value
