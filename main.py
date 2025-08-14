import logging
import sys

from  process import VideoProcessor

class CustomFormatter(logging.Formatter):

    blue = "\x1b[36m"
    grey = "\x1b[1;30m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


if __name__ == '__main__':
    root = logging.getLogger()
    LOG_LVL = logging.DEBUG

    root.setLevel(LOG_LVL)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOG_LVL)
    formatter = logging.Formatter("")
    handler.setFormatter(CustomFormatter())
    root.addHandler(handler)


    root.info('Iniciando App.')
    VideoProcessor(root)

