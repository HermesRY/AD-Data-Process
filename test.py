import argparse
import logging
from dataset import AlzheimerDataset

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help='Subject index')
args = parser.parse_args()

logging.basicConfig(filename=f"{args.id}.log", format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')
logger.setLevel(level=logging.INFO)

data_path = f"/mnt/nas/{args.id}/data"
save_path = f"./{args.id}"

ad = AlzheimerDataset(root=data_path, target_path=save_path, logger=logger)

ad.run()
