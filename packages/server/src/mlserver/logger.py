import logging

from rich.logging import RichHandler

logger = logging.getLogger("mlserve")
logger.setLevel(logging.DEBUG)

logger.addHandler(RichHandler(show_path=False))

file_handler = logging.FileHandler("mlserve.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
