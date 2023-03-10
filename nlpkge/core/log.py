import logging.config

def save_logger(logfile_path="../dataset/nlpkge.log",rank=-1):

    standard_format = '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]' \
                    '[%(levelname)s][%(message)s]'
    # simple_format = '[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]'
    simple_format = '[%(asctime)s] - [%(message)s]'
    LOGGING_DIC = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': standard_format
            },
            'simple': {
                'format': simple_format
            },
        },
        'filters': {},
        'handlers': {

            'stream': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            },

            'file': {

                'level': 20,
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': None,
                'maxBytes': 1024 * 1024 * 5,
                'backupCount': 5,
                'encoding': 'utf-8',
            },
        },

        'loggers': {
            '': {
                'handlers': ['stream', 'file'],
                'level': 'INFO',
                'propagate': True,
            },
        },
    }

    LOGGING_DIC['loggers']['']['level'] = 'INFO' if rank in [-1,0] else 'WARN'
    LOGGING_DIC['handlers']['file']['filename'] = logfile_path
    if rank not in [-1,0]:
        # 其余进程仅向stream中打印Log,不创建文件
        LOGGING_DIC['loggers']['']['handlers'] = ['stream']
        del LOGGING_DIC['handlers']['file']
    logging.config.dictConfig(LOGGING_DIC)
    logger = logging.getLogger(__name__)
    return logger


# logger = save_logger()
