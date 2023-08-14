import argparse
import logging
from nx_samplers import NxAlzheimerDataset
from rpi_samplers import RpiAlzheimerDataset

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help='Subject index')
parser.add_argument('--env', default='nx', type=str, help='nx or rpi')
parser.add_argument('--workers', type=int, default=32, help='number of workers')
args = parser.parse_args()

logging.basicConfig(filename=f"{args.id}.log", format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')
logger.setLevel(level=logging.INFO)

save_path = f"/pm1733_x3/sample_li/{args.id}"

if args.env == 'nx':
    data_path = f"/mnt/hdd_nas/AD-Data/{args.id}/data"
    if args.id.startswith("2-NX"):
        data_path = f"/pm1733_x3/AD-Data/{args.id}/data"
    ad = NxAlzheimerDataset(root=data_path, target_path=save_path, logger=logger, num_workers=args.workers)
elif args.env == 'rpi':
    data_path = f"/mnt/hdd_nas/AD-Data/{args.id}"
    ad = RpiAlzheimerDataset(root=data_path, target_path=save_path, logger=logger, num_workers=args.workers)

ad.run()
