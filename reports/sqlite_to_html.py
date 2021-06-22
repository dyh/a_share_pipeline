import time

import config
from utils.date_time import get_week_day
from utils.sqlite import SQLite

if __name__ == '__main__':
    # 获取要输出到html的tic
    config.SINGLE_A_STOCK_CODE = ['sh.600036', ]

    # 交易详情
    text_trade_detail = ''

    # 从 index.html.template 文件中 读取 网页模板
    with open('./template/index.html.template', 'r') as index_page_template:
        text_index_page_template = index_page_template.read()

        all_text_card = ''

        # 从 card_predict_result.template 文件中 读取 结果卡片 模板
        with open('./template/card_predict_result.template', 'r') as result_card_template:
            text_card = result_card_template.read()

            db_path = '../stock_db/stock.db'

            # 连接数据库
            sqlite = SQLite(db_path)

            for tic in config.SINGLE_A_STOCK_CODE:

                table_name = tic + '_report'

                # 复制一个结果卡片模板，将以下信息 填充到卡片中
                copy_text_card = text_card

                # 获取数据库中最大hold，最大action，用于计算百分比
                sql_cmd = f'SELECT abs("hold") FROM "{table_name}" ORDER BY abs("hold") DESC LIMIT 1'
                max_hold = sqlite.fetchone(sql_cmd)
                max_hold = max_hold[0]

                # max_action
                sql_cmd = f'SELECT abs("action") FROM "{table_name}" ORDER BY abs("action") DESC LIMIT 1'
                max_action = sqlite.fetchone(sql_cmd)
                max_action = max_action[0]

                # 获取数据库中最大日期
                sql_cmd = f'SELECT "date" FROM "{table_name}" ORDER BY "date" DESC LIMIT 1'
                max_date = sqlite.fetchone(sql_cmd)
                max_date = str(max_date[0])

                # 用此最大日期查询出一批数据
                sql_cmd = f'SELECT "id", "agent", "vali_period_value", "pred_period_name", "action", "hold", "day", "episode_return", "max_return", "trade_detail" ' \
                          f'FROM "{table_name}" WHERE "date" = \'{max_date}\' ORDER BY episode_return DESC'

                list_result = sqlite.fetchall(sql_cmd)

                text_table_tr_td = ''

                for item_result in list_result:
                    # 替换字符串内容
                    copy_text_card = copy_text_card.replace('<%tic%>', tic)
                    copy_text_card = copy_text_card.replace('<%tic_no_dot%>', tic.replace('.', ''))

                    id1, agent1, vali_period_value1, pred_period_name1, action1, hold1, day1, episode_return1, max_return1, trade_detail1 = item_result

                    # <%day%>
                    copy_text_card = copy_text_card.replace('<%day%>', day1)

                    # 改为百分比
                    action1 = float(action1)
                    max_action = float(max_action)
                    hold1 = float(hold1)
                    max_hold = float(max_hold)
                    episode_return1 = float(episode_return1)
                    max_return1 = float(max_return1)

                    action1 = round(action1 * 100 / max_action, 0)
                    hold1 = round(hold1 * 100 / max_hold, 0)
                    # agent1
                    agent1 = agent1[5:]

                    # 回报
                    episode_return1 = round((episode_return1 - 1) * 100, 2)
                    max_return1 = round((max_return1 - 1) * 100, 2)

                    text_table_tr_td += f'<tr>' \
                                        f'<td>{episode_return1}% / {max_return1}%</td>' \
                                        f'<td>{action1}%</td>' \
                                        f'<td>{hold1}%</td>' \
                                        f'<td>{agent1}</td>' \
                                        f'<td>{vali_period_value1}天</td>' \
                                        f'<td>第{day1}/{pred_period_name1}天</td>' \
                                        f'</tr>'

                    # 交易详情，trade_detail1，保存为独立文件

                    text_trade_detail += f'\r\n{"-" * 20} {agent1} {vali_period_value1}天 {"-" * 20}\r\n'

                    text_trade_detail += f'{episode_return1}% / {max_return1}% ' \
                                         f' {action1}% ' \
                                         f' {hold1}% ' \
                                         f' {agent1} ' \
                                         f' {vali_period_value1}天 ' \
                                         f' 第{day1}/{pred_period_name1}天\r\n'

                    text_trade_detail += '\r\n交易详情\r\n\r\n'

                    text_trade_detail += trade_detail1

                    pass
                pass

                # 日期
                date1 = max_date + ' ' + get_week_day(max_date)
                copy_text_card = copy_text_card.replace('<%date%>', date1)

                # 按 hold 分组，选出数量最多的 hold
                sql_cmd = f'SELECT "hold", COUNT(id) as count1 ' \
                          f'FROM "{table_name}" WHERE "date" = \'{max_date}\' GROUP BY "hold"' \
                          f' ORDER BY count1 DESC, abs("hold") DESC LIMIT 1'

                most_hold = sqlite.fetchone(sql_cmd)[0]
                most_hold = float(most_hold)
                most_hold = round(most_hold * 100 / max_hold, 0)
                copy_text_card = copy_text_card.replace('<%most_hold%>', str(most_hold))

                # 按 action 分组，取数量最多的 action
                sql_cmd = f'SELECT "action", COUNT("id") as count1 ' \
                          f'FROM "{table_name}" WHERE "date" = \'{max_date}\' GROUP BY "action"' \
                          f' ORDER BY count1 DESC, abs("action") DESC LIMIT 1'

                most_action = sqlite.fetchone(sql_cmd)[0]
                most_action = float(most_action)
                most_action = round(most_action * 100 / max_action, 0)
                copy_text_card = copy_text_card.replace('<%most_action%>', str(most_action))

                # 表格
                all_text_card += copy_text_card.replace('<%predict_result_table_tr_td%>', text_table_tr_td)
                all_text_card += '\r\n'

                if text_trade_detail is not '':
                    # 写入交易详情文件
                    with open(f'./{tic}.txt', 'w') as file_detail:
                        file_detail.write(text_trade_detail)
                        pass
                    pass
                pass
            pass

            sqlite.close()
        pass

        # 将多个 卡片模板 替换到 网页模板
        text_index_page_template = text_index_page_template.replace('<%page_content%>', all_text_card)

        current_time_point = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text_index_page_template = text_index_page_template.replace('<%page_time_point%>', current_time_point)

        text_index_page_template = text_index_page_template.replace('<%page_title%>', 'A股预测')

        INDEX_HTML_PAGE_PATH = './index.html'

        # 写入网页文件
        with open(INDEX_HTML_PAGE_PATH, 'w') as file_index:
            file_index.write(text_index_page_template)
            pass
        pass

    pass
