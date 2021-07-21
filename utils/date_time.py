import datetime
import time


def time_point(time_format='%Y%m%d_%H%M%S'):
    return time.strftime(time_format, time.localtime())


def get_datetime_from_date_str(str_date):
    """
    由日期字符串获得datetime对象
    :param str_date: 日期字符串，格式 2021-10-20
    :return: datetime对象
    """
    year_temp = str_date.split('-')[0]
    month_temp = str_date.split('-')[1]
    day_temp = str_date.split('-')[2]
    return datetime.date(int(year_temp), int(month_temp), int(day_temp))


# def get_next_work_day(datetime_date, next_flag=1):
#     """
#     获取下一个工作日
#     :param datetime_date: 日期类型
#     :param next_flag: -1 前一天，+1 后一天
#     :return: datetime.date
#     """
#     if next_flag > 0:
#         next_date = datetime_date + datetime.timedelta(
#             days=7 - datetime_date.weekday() if datetime_date.weekday() > 3 else 1)
#     else:
#         # 周二至周六
#         if 0 < datetime_date.weekday() < 6:
#             next_date = datetime_date - datetime.timedelta(days=1)
#             pass
#         elif 0 == datetime_date.weekday():
#             # 周一
#             next_date = datetime_date - datetime.timedelta(days=3)
#         else:
#             # 6 == datetime_date.weekday():
#             # 周日
#             next_date = datetime_date - datetime.timedelta(days=2)
#             pass
#         pass
#     return next_date

def get_next_work_day(datetime_date, next_flag=1):
    """
    获取下一个工作日
    :param datetime_date: 日期类型
    :param next_flag: -1 前一天，+1 后一天
    :return: datetime.date
    """
    for loop in range(next_flag.__abs__()):
        if next_flag > 0:
            datetime_date = datetime_date + datetime.timedelta(
                days=7 - datetime_date.weekday() if datetime_date.weekday() > 3 else 1)
        else:
            # 周二至周六
            if 0 < datetime_date.weekday() < 6:
                datetime_date = datetime_date - datetime.timedelta(days=1)
                pass
            elif 0 == datetime_date.weekday():
                # 周一
                datetime_date = datetime_date - datetime.timedelta(days=3)
            else:
                # 6 == datetime_date.weekday():
                # 周日
                datetime_date = datetime_date - datetime.timedelta(days=2)
                pass
            pass
    return datetime_date


def get_next_day(datetime_date, next_flag=1):
    """
    获取下一个日期
    :param datetime_date: 日期类型
    :param next_flag: -2 前2天，+1 后1天
    :return: datetime.date
    """
    if next_flag > 0:
        next_date = datetime_date + datetime.timedelta(days=abs(next_flag))
    else:
        next_date = datetime_date - datetime.timedelta(days=abs(next_flag))
    return next_date


def get_today_date():
    # 获取今天日期
    time_format = '%Y-%m-%d'
    return time.strftime(time_format, time.localtime())


def is_greater(date1, date2):
    """
    日期1是否大于日期2
    :param date1: 日期1字符串 2001-03-01
    :param date2: 日期2字符串 2001-01-01
    :return: True/Flase
    """
    temp1 = time.strptime(date1, '%Y-%m-%d')
    temp2 = time.strptime(date2, '%Y-%m-%d')

    if temp1 > temp2:
        return True
    else:
        return False
    pass


def get_begin_vali_date_list(end_vali_date):
    """
    获取7个日期列表
    :return: list()
    """
    list_result = list()
    # for work_days in [20, 30, 40, 50, 60, 72, 90, 100, 150, 200, 300, 500, 518, 1000, 1200, 1268]:
    for work_days in [30, 40, 50, 60, 72, 90, 100, 150, 200, 300, 500, 518, 1000, 1200]:
        begin_vali_date = get_next_work_day(end_vali_date, next_flag=-work_days)
        list_result.append((work_days, begin_vali_date))

    list_result.reverse()

    return list_result


def get_end_vali_date_list(begin_vali_date):
    """
    获取7个日期列表
    :return: list()
    """

    list_result = list()

    # # 20周期
    # end_vali_date = get_next_day(begin_vali_date, next_flag=28)
    # list_result.append((20, end_vali_date))
    #
    # # 30周期
    # end_vali_date = get_next_day(begin_vali_date, next_flag=42)
    # list_result.append((30, end_vali_date))
    #
    # # 40周期
    # end_vali_date = get_next_day(begin_vali_date, next_flag=56)
    # list_result.append((40, end_vali_date))
    #
    # # 50周期
    # end_vali_date = get_next_day(begin_vali_date, next_flag=77)
    # list_result.append((50, end_vali_date))
    #
    # # 60周期
    # end_vali_date = get_next_day(begin_vali_date, next_flag=91)
    # list_result.append((60, end_vali_date))
    #
    # # 72周期
    # end_vali_date = get_next_day(begin_vali_date, next_flag=108)
    # list_result.append((72, end_vali_date))
    #
    # # 90周期
    # end_vali_date = get_next_day(begin_vali_date, next_flag=134)
    # list_result.append((90, end_vali_date))

    # 50周期
    work_days = 50
    end_vali_date = get_next_work_day(begin_vali_date, next_flag=work_days)
    list_result.append((work_days, end_vali_date))

    # 100周期
    work_days = 100
    end_vali_date = get_next_work_day(begin_vali_date, next_flag=work_days)
    list_result.append((work_days, end_vali_date))

    # 150周期
    work_days = 150
    end_vali_date = get_next_work_day(begin_vali_date, next_flag=work_days)
    list_result.append((work_days, end_vali_date))

    # 200周期
    work_days = 200
    end_vali_date = get_next_work_day(begin_vali_date, next_flag=work_days)
    list_result.append((work_days, end_vali_date))

    # 300周期
    work_days = 300
    end_vali_date = get_next_work_day(begin_vali_date, next_flag=work_days)
    list_result.append((work_days, end_vali_date))

    # 500周期
    work_days = 500
    end_vali_date = get_next_work_day(begin_vali_date, next_flag=work_days)
    list_result.append((work_days, end_vali_date))

    # 1000周期
    work_days = 1000
    end_vali_date = get_next_work_day(begin_vali_date, next_flag=work_days)
    list_result.append((work_days, end_vali_date))

    return list_result


def get_week_day(string_time_point):
    week_day_dict = {
        0: '周一',
        1: '周二',
        2: '周三',
        3: '周四',
        4: '周五',
        5: '周六',
        6: '周日',
    }

    day_of_week = datetime.datetime.fromtimestamp(
        time.mktime(time.strptime(string_time_point, "%Y-%m-%d"))).weekday()

    return week_day_dict[day_of_week]
