import os
import librosa
import soundfile
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime, timedelta


class DepthSampler:
    def __init__(self, root):
        self.root = root
        self._folder_navigation()

    def _folder_navigation(self):
        depth_video = [file for file in os.listdir(self.root) if file.endswith('.avi')]
        depth_csv = [file for file in os.listdir(self.root) if file.endswith('.csv')]
        dataframes = [pd.read_csv(os.path.join(self.root, file)) for file in depth_csv]
        # timestamps with microseconds
        end_timestamp = [df['timestamp'].iloc[-1] for df in dataframes]
        self.filenames = depth_video
        # timestamps without microseconds
        self.end_timestamp = [self._drop_microseconds(string) for string in end_timestamp]
        self.end_time = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in self.end_timestamp]
        self.start_timestamps = [os.path.splitext(file)[0] for file in depth_video]
        self.start_time = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in self.start_timestamps]
        self.durations = [(end - start).total_seconds() for end, start in zip(self.end_time, self.start_time)]
        self.total_duration = sum(self.durations)
        del depth_video
        del depth_csv
        del dataframes
        del end_timestamp

    @staticmethod
    def _drop_microseconds(string):
        """
        Load the string (format = %Y-%m-%d_%H-%M-%S.%f) to %Y-%m-%d_%H-%M-%S, i.e., drop the microseconds.
        """
        str_time = datetime.strptime(string, '%Y-%m-%d_%H-%M-%S.%f')
        str_wo_ms = str_time.strftime('%Y-%m-%d_%H-%M-%S')
        return str_wo_ms

