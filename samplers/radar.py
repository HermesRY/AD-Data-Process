import os
import pickle
import librosa
import soundfile
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime, timedelta


class RadarSampler:
    def __init__(self, root):
        self.root = root
        self._folder_navigation()

    def _folder_navigation(self):
        radar_files = [file for file in os.listdir(self.root) if file.endswith('.pkl')]
        csv_files = [self._read_pkl_as_csv(file) for file in radar_files]
        end_timestamps = [df['Time'].iloc[-1] for df in csv_files]
        self.filenames = radar_files
        self.start_timestamps = [os.path.splitext(file)[0] for file in radar_files]
        self.start_time = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in self.start_timestamps]
        self.end_timestamps = [self._drop_milliseconds(dt_str) for dt_str in end_timestamps]
        self.end_time = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in self.end_timestamps]
        self.durations = [(end - start).total_seconds() for end, start in zip(self.end_time, self.start_time)]
        self.total_duration = sum(self.durations)
        del radar_files
        del csv_files
        del end_timestamps

    def _read_pkl_as_csv(self, filename):
        with open(os.path.join(self.root, filename), 'rb') as file:
            data = pickle.load(file)
        return pd.DataFrame(data)

    @staticmethod
    def _drop_milliseconds(string):
        str_time = datetime.strptime(string, '%Y%m%d-%H%M%S-%f')
        str_wo_ms = str_time.strftime('%Y-%m-%d_%H-%M-%S')
        return str_wo_ms

