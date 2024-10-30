from time import localtime

CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"
BEST_MODEL_SAVE = "./best-model/"
BEST_MODEL_LOGS = "./best-model-logs/"


def get_current_time():
    return "{}{}{}{}{}".format(
        localtime().tm_year,
        localtime().tm_mon,
        localtime().tm_mday,
        localtime().tm_hour,
        localtime().tm_min,
    )
