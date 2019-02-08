import logging
import sys
import os
import subprocess
from . import utils


def get_logger(fname='', debug=True, dirname='./'):
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
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        f_handler = logging.FileHandler('%s/%s.log'%(utils.format_dirname(dirname),fname),mode='w')
        f_handler.setFormatter(log_format)
        logger.addHandler(f_handler)

    logger.addHandler(c_handler)
    logger.info("Running code version: [%s]" %get_git_revision_short_hash())
    return logger



def get_git_revision_short_hash():
    version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return version.decode('utf-8')
