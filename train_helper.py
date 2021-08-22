import config
from utils import date_time
from utils.date_time import get_next_work_day
from utils.sqlite import SQLite


def init_model_hyper_parameters_table_sqlite():
    # 初始化模型超参表 model_hyper_parameters 和 训练历史记录表 train_history

    time_point = date_time.time_point(time_format='%Y-%m-%d %H:%M:%S')

    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'model_hyper_parameters'

    sqlite = SQLite(db_path)

    if_exists = sqlite.table_exists(table_name)

    if if_exists is None:
        # 如果是初始化，则创建表
        sqlite.execute_non_query(sql=f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                                     f'model_name TEXT NOT NULL UNIQUE, if_on_policy INTEGER NOT NULL, '
                                     f'break_step INTEGER NOT NULL, train_reward_scale INTEGER NOT NULL, '
                                     f'eval_reward_scale INTEGER NOT NULL, training_times INTEGER NOT NULL, '
                                     f'state_amount_scale INTEGER NOT NULL, state_price_scale INTEGER NOT NULL, '
                                     f'state_stocks_scale INTEGER NOT NULL, state_tech_scale INTEGER NOT NULL, '
                                     f'time_point TEXT NOT NULL, if_active INTEGER NOT NULL);')

        # 提交
        sqlite.commit()
        pass

        # 初始化默认值
        for agent_item in ['AgentSAC', 'AgentPPO', 'AgentTD3', 'AgentDDPG', 'AgentModSAC', ]:

            if agent_item == 'AgentPPO':
                if_on_policy = 1
                break_step = 50000
            elif agent_item == 'AgentModSAC':
                if_on_policy = 0
                break_step = 50000
            else:
                if_on_policy = 0
                break_step = 5000
            pass

            # 例如 2 ** -6 这里只将 -6 保存进数据库
            train_reward_scale = -12
            eval_reward_scale = -8

            # 训练次数
            training_times = 0

            state_amount_scale = -25
            state_price_scale = -9
            state_stocks_scale = -16
            state_tech_scale = -9

            for work_days in [300, ]:
                # 如果是初始化，则创建表
                sql_cmd = f'INSERT INTO "{table_name}" (model_name, if_on_policy, break_step, ' \
                          f'train_reward_scale, eval_reward_scale, ' \
                          f'state_amount_scale, state_price_scale, state_stocks_scale, state_tech_scale,' \
                          f'training_times, time_point, if_active) ' \
                          f'VALUES (?,?,?,?,?,?,?,?,?,?,?,1)'

                sql_values = (agent_item + '_' + str(work_days), if_on_policy, break_step,
                              train_reward_scale, eval_reward_scale,
                              state_amount_scale, state_price_scale, state_stocks_scale, state_tech_scale,
                              training_times, time_point)

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
                                     f'model_id TEXT NOT NULL, '
                                     f'train_reward_value NUMERIC NOT NULL, eval_reward_value NUMERIC NOT NULL, '
                                     f'state_amount_value NUMERIC NOT NULL, state_price_value NUMERIC NOT NULL, '
                                     f'state_stocks_value NUMERIC NOT NULL, state_tech_value NUMERIC NOT NULL, '
                                     f'time_point TEXT NOT NULL);')

        # 提交
        sqlite.commit()
        pass
    pass

    sqlite.close()

    pass


def query_model_hyper_parameters_sqlite(model_name=None):
    # 根据 model_name 查询模型超参

    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'model_hyper_parameters'

    sqlite = SQLite(db_path)

    # 'state_amount_scale INTEGER NOT NULL, state_price_scale INTEGER NOT NULL, '
    # 'state_stocks_scale INTEGER NOT NULL, state_tech_scale INTEGER NOT NULL, '

    if model_name is None:
        query_sql = f'SELECT id, model_name, if_on_policy, break_step, train_reward_scale, eval_reward_scale, ' \
                    f'training_times, time_point, state_amount_scale, state_price_scale, state_stocks_scale, ' \
                    f'state_tech_scale FROM "{table_name}" WHERE if_active=1 ' \
                    f' ORDER BY training_times ASC LIMIT 1'
    else:
        # 唯一记录
        query_sql = f'SELECT id, model_name, if_on_policy, break_step, train_reward_scale, eval_reward_scale, ' \
                    f'training_times, time_point, state_amount_scale, state_price_scale, state_stocks_scale, ' \
                    f'state_tech_scale FROM "{table_name}" WHERE model_name="{model_name}"' \
                    f' LIMIT 1'
    pass

    id1, model_name, if_on_policy, break_step, train_reward_scale, eval_reward_scale, training_times, time_point, \
    state_amount_scale, state_price_scale, state_stocks_scale, state_tech_scale = sqlite.fetchone(query_sql)

    sqlite.close()

    return id1, model_name, if_on_policy, break_step, train_reward_scale, eval_reward_scale, training_times, \
           time_point, state_amount_scale, state_price_scale, state_stocks_scale, state_tech_scale


def update_model_hyper_parameters_table_sqlite(model_hyper_parameters_id, train_reward_scale, eval_reward_scale,
                                               training_times, state_amount_scale, state_price_scale,
                                               state_stocks_scale, state_tech_scale):
    time_point = date_time.time_point(time_format='%Y-%m-%d %H:%M:%S')

    # 更新超参表
    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'model_hyper_parameters'

    sqlite = SQLite(db_path)

    # 如果是初始化，则创建表
    sqlite.execute_non_query(sql=f'UPDATE "{table_name}" SET train_reward_scale={train_reward_scale}, '
                                 f'eval_reward_scale={eval_reward_scale}, training_times={training_times}, '
                                 f'time_point="{time_point}", state_amount_scale={state_amount_scale}, '
                                 f'state_price_scale={state_price_scale}, state_stocks_scale={state_stocks_scale}, '
                                 f'state_tech_scale={state_tech_scale} WHERE id={model_hyper_parameters_id}')
    # 提交
    sqlite.commit()

    sqlite.close()
    pass


def clear_train_history_table_sqlite():
    # 清空训练历史记录表
    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'train_history'

    sqlite = SQLite(db_path)

    sqlite.execute_non_query(sql=f'DELETE FROM "{table_name}"')
    # 提交
    sqlite.commit()
    pass

    sqlite.close()
    pass


def insert_train_history_record_sqlite(model_id, train_reward_value=0.0, eval_reward_value=0.0,
                                       state_amount_value=0.0, state_price_value=0.0, state_stocks_value=0.0,
                                       state_tech_value=0.0):
    time_point = date_time.time_point(time_format='%Y-%m-%d %H:%M:%S')

    # 插入训练历史记录
    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'train_history'

    sqlite = SQLite(db_path)

    sql_cmd = f'INSERT INTO "{table_name}" ' \
              f'(model_id, train_reward_value, eval_reward_value, time_point, ' \
              f'state_amount_value, state_price_value, state_stocks_value, state_tech_value) VALUES (?,?,?,?,?,?,?,?);'

    sql_values = (model_id, train_reward_value, eval_reward_value, time_point,
                  state_amount_value, state_price_value, state_stocks_value, state_tech_value)

    sqlite.execute_non_query(sql_cmd, sql_values)

    # 提交
    sqlite.commit()

    sqlite.close()

    pass


def loop_scale_one(max_state_value, origin_state_scale):
    if max_state_value is None:
        new_state_scale = origin_state_scale
    else:
        i = 0
        while max_state_value >= 1.0:
            max_state_value = max_state_value / 2
            i += 1
        pass
        new_state_scale = origin_state_scale - i
    pass

    return new_state_scale


def update_model_hyper_parameters_by_train_history(model_hyper_parameters_id, origin_train_reward_scale,
                                                   origin_eval_reward_scale, origin_training_times,
                                                   origin_state_amount_scale, origin_state_price_scale,
                                                   origin_state_stocks_scale, origin_state_tech_scale):
    # 根据reward历史，更新超参表
    # 插入训练历史记录
    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'train_history'

    sqlite = SQLite(db_path)

    query_sql = f'SELECT MAX(train_reward_value), MAX(eval_reward_value), ' \
                f'MAX(state_amount_value), MAX(state_price_value), ' \
                f'MAX(state_stocks_value), MAX(state_tech_value)  FROM "{table_name}" ' \
                f' WHERE model_id="{model_hyper_parameters_id}"'

    max_train_reward_value, max_eval_reward_value, max_state_amount_value, max_state_price_value, \
    max_state_stocks_value, max_state_tech_value = sqlite.fetchone(query_sql)

    sqlite.close()

    # reward 阈值
    reward_threshold = config.REWARD_THRESHOLD

    if max_train_reward_value is None:
        new_train_reward_scale = origin_train_reward_scale
        print('> keep origin train_reward_scale', new_train_reward_scale)
        pass
    else:
        if max_train_reward_value >= reward_threshold:
            new_train_reward_scale = origin_train_reward_scale - (max_train_reward_value // reward_threshold)
            print('> modify train_reward_scale:', origin_train_reward_scale, '->', new_train_reward_scale)
        else:
            new_train_reward_scale = origin_train_reward_scale
            print('> keep origin train_reward_scale', new_train_reward_scale)
        pass
    pass

    if max_eval_reward_value is None:
        new_eval_reward_scale = origin_eval_reward_scale
        print('> keep origin eval_reward_scale', new_eval_reward_scale)
        pass
    else:
        if max_eval_reward_value >= reward_threshold:
            new_eval_reward_scale = origin_eval_reward_scale - (max_eval_reward_value // reward_threshold)

            print('> modify eval_reward_scale:', origin_eval_reward_scale, '->', new_eval_reward_scale)
            pass
        else:
            new_eval_reward_scale = origin_eval_reward_scale

            print('> keep origin eval_reward_scale', new_eval_reward_scale)
            pass
        pass
    pass

    new_state_amount_scale = loop_scale_one(max_state_amount_value, origin_state_amount_scale)
    if new_state_amount_scale == origin_state_amount_scale:
        print('> keep origin state_amount_scale', new_state_amount_scale)
    else:
        print('> modify state_amount_scale:', origin_state_amount_scale, '->', new_state_amount_scale)
    pass

    new_state_price_scale = loop_scale_one(max_state_price_value, origin_state_price_scale)
    if new_state_price_scale == origin_state_price_scale:
        print('> keep origin state_price_scale', new_state_price_scale)
    else:
        print('> modify state_price_scale:', origin_state_price_scale, '->', new_state_price_scale)
    pass

    new_state_stocks_scale = loop_scale_one(max_state_stocks_value, origin_state_stocks_scale)
    if new_state_stocks_scale == origin_state_stocks_scale:
        print('> keep origin state_stocks_scale', new_state_stocks_scale)
    else:
        print('> modify state_stocks_scale:', origin_state_stocks_scale, '->', new_state_stocks_scale)
    pass

    new_state_tech_scale = loop_scale_one(max_state_tech_value, origin_state_tech_scale)
    if new_state_tech_scale == origin_state_tech_scale:
        print('> keep origin state_tech_scale', new_state_tech_scale)
    else:
        print('> modify state_tech_scale:', origin_state_tech_scale, '->', new_state_tech_scale)
    pass

    # 更新超参表
    update_model_hyper_parameters_table_sqlite(model_hyper_parameters_id=model_hyper_parameters_id,
                                               train_reward_scale=new_train_reward_scale,
                                               eval_reward_scale=new_eval_reward_scale,
                                               training_times=origin_training_times + 1,
                                               state_amount_scale=new_state_amount_scale,
                                               state_price_scale=new_state_price_scale,
                                               state_stocks_scale=new_state_stocks_scale,
                                               state_tech_scale=new_state_tech_scale)

    pass


def query_begin_vali_date_list_by_agent_name(agent_name, end_vali_date):
    list_result = list()

    # 连接数据库
    db_path = config.STOCK_DB_PATH

    # 表名
    table_name = 'model_hyper_parameters'

    sqlite = SQLite(db_path)

    query_sql = f'SELECT model_name FROM "{table_name}" WHERE if_active=1 AND model_name LIKE "{agent_name}%" ' \
                f' ORDER BY model_name ASC'

    list_temp = sqlite.fetchall(query_sql)

    sqlite.close()

    for work_days in list_temp:
        # AgentSAC_60 --> 60
        work_days = int(str(work_days[0]).split('_')[1])
        begin_vali_date = get_next_work_day(end_vali_date, next_flag=-work_days)
        list_result.append((work_days, begin_vali_date))
    pass

    list_temp.clear()

    return list_result
