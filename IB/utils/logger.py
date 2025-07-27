import colorlog
import logging

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "{asctime}.{msecs:03.0f} [{log_color}{levelname}{reset}] {filename}:{lineno} - {funcName}() - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []  # Optional: Clear other handlers
logger.addHandler(handler)
logger.propagate = False
