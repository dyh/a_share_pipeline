
# current work directory，正在训练使用的目录，cwd: ./AgentPPO/StockTradingEnv-v1_0
# from ctypes import c_char_p
# from multiprocessing import Manager
# manager = Manager()
# CWD = manager.Value(c_char_p, "")

# CWD = './AgentPPO/StockTradingEnv-v1'
CWD = ''

AGENT_NAME = ''

# 单支股票代码List
SINGLE_A_STOCK_CODE = []

# 显示预测信息
IF_SHOW_PREDICT_INFO = True

# 工作日标记，用于加载对应的weights
WORK_DAY_FLAG = ''

## time_fmt = '%Y-%m-%d'
START_DATE = ""
START_EVAL_DATE = ""
END_DATE = ""
# 要输出的日期
OUTPUT_DATE = ''

DATA_SAVE_DIR = f"datasets"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"
LOGGER_DIR = f"logger_log"

# batch股票数据库地址
STOCK_DB_PATH = "./" + DATA_SAVE_DIR + '/stock.db'

# 批量训练股票代码List
BATCH_A_STOCK_CODE = []

HS300_CODE_LIST = []

# 沪深300数据库地址
HS300_DB_PATH = "./" + DATA_SAVE_DIR + '/hs300.db'

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

## Model Parameters
# A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.0002}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.0002}

# DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
# TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

# SAC_PARAMS = {
#     "batch_size": 64,
#     "buffer_size": 100000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1",
# }
