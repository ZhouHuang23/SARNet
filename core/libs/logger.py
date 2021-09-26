# -*- coding: utf-8 -*-

import logging
import colorlog

log_colors_config = {
    'DEBUG': 'white',
    'INFO': 'purple',
    'WARNING': 'blue',
    'ERROR': 'yellow',
    'CRITICAL': 'bold_red',
}


def set_logger(logs_filename=None):
    global file_handler
    file_handler = None

    logger = logging.getLogger('logger_name')
    # Output to console
    console_handler = logging.StreamHandler()

    # Output to file
    if logs_filename is not None:
        file_handler = logging.FileHandler(filename=logs_filename, mode='a', encoding='utf8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s.%(msecs)03d]%(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

    # log level
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # Log output format
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s# %(message)s',
        log_colors=log_colors_config
    )
    console_handler.setFormatter(console_formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        if logs_filename is not None:
            logger.addHandler(file_handler)

    console_handler.close()
    if logs_filename is not None:
        file_handler.close()

    return logger


if __name__ == '__main__':
    logger = set_logger()
    logger.debug('hello')
    logger.info('hello')
    logger.warning('hello')
    logger.error('hello')
