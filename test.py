import argparse
import logging
from dataset import AlzheimerDataset

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help='Subject index')
parser.add_argument('--workers', type=int, default=16, help='number of workers')
args = parser.parse_args()

logging.basicConfig(filename=f"{args.id}.log", format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')
logger.setLevel(level=logging.INFO)

data_path = f"/mnt/nas/{args.id}/data"
save_path = f"/mnt/AD-temp-data/sample_li/{args.id}"

ad = AlzheimerDataset(root=data_path, target_path=save_path, logger=logger, num_workers=args.workers)

ad.run()
