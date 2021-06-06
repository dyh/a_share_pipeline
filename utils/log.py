import logging


def get_logger(log_file_path, log_level=logging.INFO):
    """
    返回 logger 实例
    :param log_file_path: 日志文件路径
    :param log_level: 日志记录级别 logging.INFO、logging.ERROR、logging.DEBUG 等等
    :return: logger 实例
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    logfile = log_file_path
    fh = logging.FileHandler(logfile, mode='a')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    fh.setLevel(log_level)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    logger.addHandler(sh)

    return logger
