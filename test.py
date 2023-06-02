import logging
from dataset import AlzheimerDataset

logging.basicConfig(filename='logger.log', format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')
logger.setLevel(level=logging.INFO)
ad = AlzheimerDataset(root="/mnt/nas/NX3/data", target_path="", logger=logger)

ad.run()
