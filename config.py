
# 交易的最小股数，可设置最低100股，即1手
MINIMUM_TRADE_SHARES = 1

# 调试 state 比例
IF_DEBUG_STATE_SCALE = False

# 调试 reward 比例
IF_DEBUG_REWARD_SCALE = True

# 多只股票代码List
BATCH_A_STOCK_CODE = []

# update reward 阈值
REWARD_THRESHOLD = 256 * 1.5

# 超参ID
MODEL_HYPER_PARAMETERS = -1

# 代理名称
AGENT_NAME = ''

# 代理预测周期
AGENT_WORK_DAY = 0

# 预测周期
PREDICT_PERIOD = ''

# 单支股票代码List
SINGLE_A_STOCK_CODE = []

# 真实的预测
IF_ACTUAL_PREDICT = False

# 工作日标记，用于加载对应的weights
# weights的vali周期
VALI_DAYS_FLAG = ''

## time_fmt = '%Y-%m-%d'
START_DATE = ''
START_EVAL_DATE = ''
END_DATE = ''
# 要输出的日期
OUTPUT_DATE = ''

DATA_SAVE_DIR = f'stock_db'

# pth路径
WEIGHTS_PATH = 'weights'

# batch股票数据库地址
STOCK_DB_PATH = './' + DATA_SAVE_DIR + '/stock.db'

# ----
# PostgreSQL
PSQL_HOST = '192.168.192.1'
PSQL_PORT = '5432'
PSQL_DATABASE = 'a_share'
PSQL_USER = 'dyh'
PSQL_PASSWORD = '9898BdlLsdfsHsbgp'
# ----

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
