import logging
import sys


def get_logger(fname=''):
    '''create logger in fname.log. '''
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    c_handler = logging.StreamHandler(sys.stderr)

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)-6s [%(filename)s:%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    c_handler.setFormatter(log_format)

    if len(fname)>0:
        f_handler = logging.FileHandler('%s.log'%(fname))
        f_handler.setFormatter(log_format)
        logger.addHandler(f_handler)

    logger.addHandler(c_handler)
    return logger
