import logging
import sys

logger = logging.getLogger('pellets')

if 'pytest' in sys.modules:
    logger.setLevel(logging.DEBUG)