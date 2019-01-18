import logging
import sys


def get_logger(fname='', debug=True):
    '''create logger in fname.log. '''
    # Create a custom logger
    logger = logging.getLogger(__name__)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler(sys.stderr)

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s [%(filename)s:%(funcName)s] %(levelname)s:: %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    c_handler.setFormatter(log_format)

    if len(fname)>0:
        f_handler = logging.FileHandler('%s.log'%(fname),mode='w')
        f_handler.setFormatter(log_format)
        logger.addHandler(f_handler)

    logger.addHandler(c_handler)
    return logger
