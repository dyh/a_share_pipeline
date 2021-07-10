import config
from utils import date_time
from utils.sqlite import SQLite


def init_model_hyper_parameters_table_sqlite():
    # 初始化模型超参表 model_hyper_parameters 和 训练历史记录表 train_history

    time_point = date_time.time_point()

    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'model_hyper_parameters'

    sqlite = SQLite(db_path)

    if_exists = sqlite.table_exists(table_name)

    if if_exists is None:
        # 如果是初始化，则创建表
        sqlite.execute_non_query(sql=f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                     f'model_name TEXT NOT NULL UNIQUE, if_on_policy TEXT NOT NULL, '
                                     f'break_step INTEGER NOT NULL, train_reward_scale INTEGER NOT NULL, '
                                     f'eval_reward_scale INTEGER NOT NULL, training_times INTEGER NOT NULL, '
                                     f'time_point TEXT NOT NULL);')
        # 提交
        sqlite.commit()
        pass

        # 初始化默认值
        for agent_item in ['AgentSAC', 'AgentPPO', 'AgentTD3', 'AgentDDPG', 'AgentModSAC', ]:

            if agent_item == 'AgentPPO':
                if_on_policy = ""
                break_step = 8e6
            elif agent_item == 'AgentModSAC':
                if_on_policy = "False"
                break_step = 3e6
            else:
                if_on_policy = "False"
                break_step = 50000
            pass

            # 例如 2 ** -6 这里只将 -6 保存进数据库
            train_reward_scale = -6
            eval_reward_scale = -5

            # 训练次数
            training_times = 0

            for work_days in [20, 30, 40, 50, 60, 72, 90, 100, 150, 200, 300, 500, 518, 1000, 1200, 1268]:
                # 如果是初始化，则创建表
                sql_cmd = f'INSERT INTO "{table_name}" (model_name, if_on_policy, break_step, ' \
                          f'train_reward_scale, eval_reward_scale, training_times,time_point) ' \
                          f'VALUES (?,?,?,?,?,?,?)'

                sql_values = (agent_item + '_' + work_days, if_on_policy, break_step, train_reward_scale,
                              eval_reward_scale, training_times, time_point)

                sqlite.execute_non_query(sql_cmd, sql_values)

                pass
            pass

        # 提交
        sqlite.commit()
    pass

    # 表名
    table_name = 'train_history'

    if_exists = sqlite.table_exists(table_name)

    if if_exists is None:
        # 如果是初始化，则创建表
        sqlite.execute_non_query(sql=f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                     f'model_id TEXT NOT NULL, train_reward_value TEXT NOT NULL, '
                                     f'eval_reward_value TEXT NOT NULL, time_point TEXT NOT NULL);')
        # 提交
        sqlite.commit()
        pass
    pass

    sqlite.close()

    pass


def query_model_hyper_parameters_sqlite():
    # 根据 model_name 查询模型超参

    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'model_hyper_parameters'

    sqlite = SQLite(db_path)

    query_sql = f'SELECT id, model_name, if_on_policy, break_step, train_reward_scale, eval_reward_scale, ' \
                f'training_times, time_point FROM "{table_name}" ' \
                f' ORDER BY training_times ASC LIMIT 1'

    id1, model_name, if_on_policy, break_step, train_reward_scale, eval_reward_scale, training_times, time_point = \
        sqlite.fetchone(query_sql)

    sqlite.close()

    return id1, model_name, if_on_policy, break_step, train_reward_scale, eval_reward_scale, training_times, time_point


def update_model_hyper_parameters_table_sqlite(id1, train_reward_scale, eval_reward_scale, training_times):
    time_point = date_time.time_point()

    # 更新超参表
    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'model_hyper_parameters'

    sqlite = SQLite(db_path)

    if_exists = sqlite.table_exists(table_name)

    if if_exists is None:
        # 如果是初始化，则创建表
        sqlite.execute_non_query(sql=f'UPDATE "{table_name}" SET train_reward_scale={train_reward_scale}, '
                                     f'eval_reward_scale={eval_reward_scale}, training_times={training_times}, '
                                     f'time_point="{time_point}" WHERE id={id1}')
        # 提交
        sqlite.commit()
        pass

    sqlite.close()
    pass


def clear_train_history_table_sqlite():
    # 清空训练历史记录表
    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'train_history'

    sqlite = SQLite(db_path)

    if_exists = sqlite.table_exists(table_name)

    if if_exists is None:
        sqlite.execute_non_query(sql=f'DELETE FROM "{table_name}";')
        # 提交
        sqlite.commit()
        pass
    pass

    sqlite.close()
    pass


def insert_train_history_record_sqlite(model_id, train_reward_value, eval_reward_value):
    time_point = date_time.time_point()

    # 插入训练历史记录
    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'train_history'

    sqlite = SQLite(db_path)

    if_exists = sqlite.table_exists(table_name)

    if if_exists is None:
        sql_cmd = f'INSERT INTO "{table_name}" ' \
                  f'(model_id, train_reward_value, eval_reward_value, time_point) VALUES (?,?,?,?);'

        sql_values = (model_id, train_reward_value, eval_reward_value, time_point)

        sqlite.execute_non_query(sql_cmd, sql_values)

        # 提交
        sqlite.commit()
        pass
    pass

    sqlite.close()

    pass
