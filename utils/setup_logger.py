import logging
import sys

def setup_logger(distributed_rank=0, filename="debug.log"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)

    # don't log results for the non-master process
    fmt = "[%(asctime)s %(levelname)s] %(message)s"

    if distributed_rank == 0:
        fmt = "[%(asctime)s %(levelname)s] %(message)s"

    elif distributed_rank == 1:
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    return logger