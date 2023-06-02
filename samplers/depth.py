import os
import cv2
import librosa
import soundfile
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime, timedelta


class DepthSampler:
    def __init__(self, root, target_path, logger, label_length, frame_size=112,
                 timestamp_tmpl="%Y-%m-%d_%H-%M-%S", num_workers=4):
        self.root = root
        self.target_path = target_path
        self.logger = logger
        self.label_length = label_length
        self.frame_size = frame_size
        self.timestamp_tmpl = timestamp_tmpl
        self.num_workers = num_workers
        self._folder_navigation()

    def _folder_navigation(self):
        depth_video = [file for file in os.listdir(self.root) if file.endswith('.avi')]
        depth_csv = [file for file in os.listdir(self.root) if file.endswith('.csv')]
        dataframes = [pd.read_csv(os.path.join(self.root, file)) for file in depth_csv]
        # timestamps with microseconds
        end_timestamp = [df['timestamp'].iloc[-1] for df in dataframes]
        # timestamps without microseconds
        end_timestamp = [self._drop_microseconds(string) for string in end_timestamp]

        self.filenames = depth_video
        self.start_timestamps = [os.path.splitext(file)[0] for file in depth_video]
        self.end_time = [datetime.strptime(ts, self.timestamp_tmpl) for ts in end_timestamp]
        self.start_time = [datetime.strptime(ts, self.timestamp_tmpl) for ts in self.start_timestamps]
        del depth_video
        del depth_csv
        del dataframes
        del end_timestamp

    def _drop_microseconds(self, string):
        """
        Load the string (format = %Y-%m-%d_%H-%M-%S.%f) to %Y-%m-%d_%H-%M-%S, i.e., drop the microseconds.
        """
        str_time = datetime.strptime(string, "%Y-%m-%d_%H-%M-%S.%f")
        str_wo_ms = str_time.strftime(self.timestamp_tmpl)
        return str_wo_ms

    def _read_single_file(self, file_timestamp, start, end, process=True):
        video_path = os.path.join(self.root, file_timestamp+".avi")
        csv_path = os.path.join(self.root, file_timestamp+".csv")

        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d_%H-%M-%S.%f")

        to_label = df[(df['timestamp'] >= start) & (df['timestamp'] <= start+self.label_length)]
        not_to_label = df[(df['timestamp'] > start+self.label_length) & (df['timestamp'] <= end)]
        to_label_idx = to_label[''].values
        not_to_label_idx = not_to_label[''].values
        label_start, label_end = to_label_idx[0], to_label_idx[-1]
        not_label_end = not_to_label_idx[-1]
        del df
        del to_label
        del not_to_label
        del to_label_idx
        del not_to_label_idx

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Failed to read video file {:s}".format(video_path))
        else:
            to_label_frames = []
            not_to_label_frames = []
            for i in range(label_start, not_label_end + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    self.logger.error("Failed to read frame {:d} from video {:s}".format(i, video_path))
                else:
                    if process:
                        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    if i <= label_end:
                        to_label_frames.append(frame)
                    else:
                        not_to_label_frames.append(frame)
            print("Depth shape(label): ", np.array(to_label_frames).shape)
            print("Depth shape: ", np.array(not_to_label_frames).shape)

    def sample(self, time_ranges):
        with Pool(self.num_workers) as pool:
            for start, end in time_ranges:
                idx = 0
                # find the target
                for i in range(len(self.start_time)):
                    if self.start_time[i] <= start < self.end_time:
                        idx = i
                        break
                if not idx:
                    self.logger.error("Failed to find sample in range {:s} and {:s} in {:s}".format(start, end, self.root))
                else:
                    file_timestamp = self.start_timestamps[idx]
                    pool.apply_async(self._read_single_file, args=(file_timestamp, start, end))
                pool.close()
                pool.join()

