import logging
import logging.config
from dataset import AlzheimerDataset

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')
ad = AlzheimerDataset(root="/mnt/nax/NX3/data", target_path="", logger=logger)

ad.check_single_hour_overlap("2022-11-15_11-00-00")
