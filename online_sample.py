import argparse
import logging
from nx_samplers import NxAlzheimerDataset

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=32, help='number of workers')
parser.add_argument('--depth_only', action='store_true', help='only depth')
args = parser.parse_args()

logging.basicConfig(filename=f"sampling.log", format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')
logger.setLevel(level=logging.INFO)

save_path = f"/mnt/ssd/sample"
data_path = f"/mnt/ssd/data"
ad = NxAlzheimerDataset(root=data_path, target_path=save_path, logger=logger,
                        num_workers=args.workers, depth_only=args.depth_only)

ad.run()
