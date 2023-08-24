import os
import cv2
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging
from multiprocessing import Pool

logging.basicConfig(filename=f"sampling.log", format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')
logger.setLevel(level=logging.INFO)

save_label_path = "/mnt/ssd/sample/label"
data_path = "/mnt/ssd/data/depth"
timestamp_tmpl = "%Y-%m-%d_%H-%M-%S"


def _write_frames_to_video(video_path, frames, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 10, (size, size))
    for frame in frames:
        video.write(frame)
    video.release()


def _drop_microseconds(string):
    """
    Load the string (format = %Y-%m-%d_%H-%M-%S.%f) to %Y-%m-%d_%H-%M-%S, i.e., drop the microseconds.
    """

    str_time = datetime.strptime(string, "%Y-%m-%d_%H-%M-%S.%f")
    str_wo_ms = str_time.strftime(timestamp_tmpl)

    return str_wo_ms


def _find_usable_timestamp(root):
    depth_video = [file for file in os.listdir(root) if file.endswith('.avi')]
    timestamp = []
    end_timestamp = []
    for item in depth_video:
        file_ts = os.path.splitext(item)[0]
        # check if the corresponding csv exists
        csv_path = os.path.join(root, file_ts + '.csv')
        if os.path.exists(csv_path):
            try:
                csv_file = pd.read_csv(csv_path)
                if csv_file.shape[0] > 0 and isinstance(csv_file['timestamp'].iloc[-1], str):
                    timestamp.append(file_ts)
                    end_timestamp.append(csv_file['timestamp'].iloc[-1])
            except pd.errors.EmptyDataError as e:
                logger.warning(f"Failed to read CSV file {csv_path}. Error message: {e}")
    return timestamp, end_timestamp


def _folder_navigation(root):
    timestamps, end_timestamp = _find_usable_timestamp(root)
    # timestamps without microseconds
    end_timestamp = [_drop_microseconds(string) for string in end_timestamp]

    start_timestamps = timestamps
    end_time = [datetime.strptime(ts, timestamp_tmpl) for ts in end_timestamp]
    start_time = [datetime.strptime(ts, timestamp_tmpl) for ts in start_timestamps]
    return start_time, end_time, start_timestamps


def _read_single_file(root, file_timestamp, start, end, process=True):
    video_path = os.path.join(root, file_timestamp+".avi")
    csv_path = os.path.join(root, file_timestamp+".csv")
    label_length = timedelta(seconds=2)

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d_%H-%M-%S.%f")

    to_label = df[(df['timestamp'] >= start) & (df['timestamp'] <= start+label_length)]
    # starring from 0
    if to_label.shape[0] == 0:
        logger.error(
            f"No to-label data available in {video_path}. Sample range: {start} -> {start + label_length}")

    to_label_idx = to_label['frame_id'].values - 1
    label_start, label_end = to_label_idx[0], to_label_idx[-1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to read video file {:s}".format(video_path))
    else:
        to_label_frames = []
        for i in range(label_start, label_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                logger.error("Failed to read frame {:d} from video {:s}".format(i, video_path))
            else:
                if process:
                    frame = cv2.resize(frame, (112, 112))
                    to_label_frames.append(frame)

        label_timestamp = start.strftime(timestamp_tmpl)
        label_path = os.path.join(save_label_path, label_timestamp + '.mp4')

        _write_frames_to_video(label_path, to_label_frames, 112)


def wrap_read_single_file(root, file_timestamp, start, end):
    try:
        _read_single_file(root, file_timestamp, start, end)
    except Exception as e:
        path = os.path.join(root, file_timestamp + '.avi')
        logger.error("Failed to read file {:s} to load the data in range {:s} -> {:s}. Error message: {:s}".
                          format(path, start.strftime(timestamp_tmpl), end.strftime(timestamp_tmpl), str(e)))


def sample(arguments, time_ranges):
    with Pool(arguments.num_workers) as pool:
        for start, end in time_ranges:
            idx = 0
            find = False
            # find the target
            for i in range(len(arguments.start_time)):
                if arguments.start_time[i] <= start < arguments.end_time[i]:
                    idx = i
                    find = True
                    break
            if not find:
                logger.error("Failed to find sample in range {:s} and {:s} in {:s}".
                                  format(start.strftime(timestamp_tmpl), end.strftime(timestamp_tmpl), arguments.root))
            else:
                file_timestamp = arguments.start_timestamps[idx]
                pool.apply(wrap_read_single_file, args=(arguments.root, file_timestamp, start, end))
        pool.close()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--depth_only', action='store_true', help='only depth')
    args = parser.parse_args()

    start_time, end_time, start_timestamps = _folder_navigation(data_path)
    args.start_time = start_time
    args.end_time = end_time
    args.start_timestamps = start_timestamps
    args.root = data_path

    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)

    for start, end in zip(start_time, end_time):
        if (end - start) > timedelta(seconds=20):
            duration = int((end - start).total_seconds())
            selected_times = [
                (
                    start + timedelta(seconds=i * 200),
                    min(start + timedelta(seconds=i * 200 + 20), end),
                )
                for i in range(duration // 200 + 2)
                if start + timedelta(seconds=i * 200) < end
                   and (
                           duration % 200 >= 20
                           or i < duration // 200
                   )
            ]
            sample(args, selected_times)

