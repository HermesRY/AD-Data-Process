import logging
from dataset import AlzheimerDataset

logging.basicConfig(filename='logger.log', level=logging.INFO)
logger = logging.getLogger('main')
ad = AlzheimerDataset(root="/mnt/nas/NX3/data", target_path="", logger=logger)

ad.check_single_hour_overlap("2022-11-15_11-00-00")
