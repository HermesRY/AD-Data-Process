import os
import pickle
import librosa
import soundfile
import numpy as np
import pandas as pd
from pathlib import Path
from .pool import Pool
from datetime import datetime, timedelta


class RadarSampler:
    def __init__(self, root, target_path, logger, label_length, timestamp_tmpl="%Y-%m-%d_%H-%M-%S", num_workers=4):
        self.root = root
        self.target_path = target_path
        self.logger = logger
        self.label_length = label_length
        self.timestamp_tmpl = timestamp_tmpl
        self.num_workers = num_workers

        self.label_path = os.path.join(self.target_path, 'label', 'radar')
        self.unlabel_path = os.path.join(self.target_path, 'unlabel', 'radar')
        self._folder_navigation()

    def _folder_navigation(self):
        starts, end_timestamps = self._find_valid_files()
        end_timestamps = [self._drop_milliseconds(dt_str) for dt_str in end_timestamps]

        self.start_timestamps = starts
        # it takes time for radar from starting to sensing
        self.start_time = [datetime.strptime(ts, self.timestamp_tmpl) + timedelta(seconds=4) for ts in self.start_timestamps]
        self.end_time = [datetime.strptime(ts, self.timestamp_tmpl) for ts in end_timestamps]
        del starts
        del end_timestamps

    def _find_valid_files(self):
        radar_files = [file for file in os.listdir(self.root) if file.endswith('.pkl')]
        starts = []
        ends = []
        for filename in radar_files:
            data = self._read_pkl_as_csv(filename)
            if data.shape[0] != 0:
                ends.append(data['Time'].iloc[-1])
                starts.append(os.path.splitext(filename)[0])
            else:
                self.logger.warning(f"Empty radar file {os.path.join(self.root, filename)}")
        return starts, ends

    def _read_pkl_as_csv(self, filename):
        try:
            with open(os.path.join(self.root, filename), 'rb') as file:
                data = pickle.load(file)
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Failed to read radar file {os.path.join(self.root, filename)}. {e}")

    def _drop_milliseconds(self, string):
        str_time = datetime.strptime(string, '%Y%m%d-%H%M%S-%f')
        str_wo_ms = str_time.strftime(self.timestamp_tmpl)
        return str_wo_ms

    def _read_single_file(self, file_timestamp, start, end):
        filename = file_timestamp + '.pkl'
        df = self._read_pkl_as_csv(filename)
        df['Time'] = pd.to_datetime(df['Time'], format="%Y%m%d-%H%M%S-%f")

        label_length = timedelta(seconds=self.label_length)

        to_label = df[(df['Time'] >= start) & (df['Time'] <= start+label_length)]
        not_to_label = df[(df['Time'] > start+label_length) & (df['Time'] <= end)]
        if to_label.shape[0] == 0:
            self.logger.error(
                f"No to-label data available in {os.path.join(self.root, filename)}. "
                f"Sample range: {start} -> {start + label_length}")
        if not_to_label.shape[0] == 0:
            self.logger.error(
                f"No not-to-label data available in {os.path.join(self.root, filename)}. "
                f"Sample range: {start + label_length} -> {end}")
        data_to_label = np.stack(to_label['Data'].apply(self.__reshape_radar).values)
        data_not_to_label = np.stack(not_to_label['Data'].apply(self.__reshape_radar).values)

        label_ts = start.strftime(self.timestamp_tmpl)
        label_path = os.path.join(self.label_path, label_ts + '.npy')
        np.save(label_path, data_to_label)

        unlabel_ts = (start + label_length).strftime(self.timestamp_tmpl)
        unlabel_path = os.path.join(self.unlabel_path, unlabel_ts + '.npy')
        np.save(unlabel_path, data_not_to_label)

        del df
        del to_label
        del not_to_label

    def wrap_read_single_file(self, file_timestamp, start, end):
        try:
            self._read_single_file(file_timestamp, start, end)
        except Exception as e:
            path = os.path.join(self.root, file_timestamp + '.pkl')
            self.logger.error("Failed to read file {:s} to load the data in range {:s} -> {:s}. Error message: {:s}".
                              format(path, start.strftime(self.timestamp_tmpl), end.strftime(self.timestamp_tmpl), str(e)))

    @staticmethod
    def __reshape_radar(data):
        """
        Reshape one-frame radar data to (2,16,32,16)
        """
        voxel_occ = np.zeros((16, 32, 16))
        voxel_vel = np.zeros((16, 32, 16))

        x = data[0]
        y = data[1]
        z = data[2]
        v = np.clip(data[3], -4, 4)

        x_loc = np.clip((x + 8) // 0.5, 0, 15).astype(int)
        y_loc = np.clip(y // 0.25, 0, 31).astype(int)
        z_loc = np.clip((z + 8) // 0.5, 0, 15).astype(int)

        voxel_occ[x_loc, y_loc, z_loc] += 1
        voxel_vel[x_loc, y_loc, z_loc] += v

        voxel_vel /= (voxel_occ + 1e-6)  # mean velocity

        voxel = np.stack((voxel_occ, voxel_vel), axis=0)  # (2, 16, 32, 16)
        return voxel

    def sample(self, time_ranges):
        with Pool(self.num_workers) as pool:
            for start, end in time_ranges:
                idx = 0
                find = False
                # find the target
                for i in range(len(self.start_time)):
                    if self.start_time[i] <= start < self.end_time[i]:
                        idx = i
                        find = True
                        break
                if not find:
                    self.logger.error("Failed to find sample in range {:s} and {:s} in {:s}"
                                      .format(start.strftime(self.timestamp_tmpl), end.strftime(self.timestamp_tmpl), self.root))
                else:
                    file_timestamp = self.start_timestamps[idx]
                    pool.apply(self.wrap_read_single_file, args=(file_timestamp, start, end))
            pool.close()
            pool.join()



