"""
   logger output
"""
import os
import logging
from logging import handlers

FORMAT = logging.Formatter('%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s: %(message)s')

CONSOLE = logging.StreamHandler()
CONSOLE.setFormatter(FORMAT)

LOG_DIR = "../log/"
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


class Logger(object):

    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'critical':logging.CRITICAL
    }

    def __init__(self, name, level='info', when='D', backCount=3, format=FORMAT):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.level_relations.get(level))
        file = handlers.TimedRotatingFileHandler(filename=LOG_DIR + 'rnn.log',
                                               when=when,
                                               backupCount=backCount,
                                               encoding='utf-8')
        #往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        file.setFormatter(format)
        self._logger.addHandler(CONSOLE)
        self._logger.addHandler(file)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)
