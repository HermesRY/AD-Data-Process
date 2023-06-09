"""
This script is to uniformly sample the AD data
"""
import os
import librosa
import soundfile
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime, timedelta


class AudioSampler:
    def __init__(self, root, target_path, logger, label_length, timestamp_tmpl="%Y-%m-%d_%H-%M-%S", num_workers=4):
        """
        This class samples a specified percentage of data from audio files under a specified folder,
        and saves the sampled data to a target path. The audio files are separated into chunks whose size
        is specified by the `chunk_size` parameter (default: 200 seconds).

        :param root: path of the audio files, generated every hour, e.g., ./data/audio/2022-11-21_10-00-00/
        :param target_path: path to store the sampled data
        :param num_workers: number of workers
        """
        self.root = root
        self.target_path = target_path
        self.logger = logger
        self.label_length = label_length
        self.num_workers = num_workers
        self.timestamp_tmpl = timestamp_tmpl

        self.label_path = os.path.join(self.target_path, 'label', 'audio')
        self.unlabel_path = os.path.join(self.target_path, 'unlabel', 'audio')
        # make the needed directories
        self._folder_navigation()

    def _folder_navigation(self):
        """
        Count the start/end time and the duration of each audio file under the root.
        """
        audio_timestamp = [os.path.splitext(file)[0] for file in os.listdir(self.root) if file.endswith('.wav')]
        csv_timestamp = [os.path.splitext(file)[0] for file in os.listdir(self.root) if file.endswith('.csv')]
        common_timestamp = [ts for ts in audio_timestamp if ts in csv_timestamp]
        audio_files = [ts + '.wav' for ts in common_timestamp]
        durations = [self._get_duration(file) for file in audio_files]
        # calculate the total durations of the audio files under the root
        self.filenames = audio_files
        # count the start timestamps
        self.start_timestamps = common_timestamp
        self.start_time = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in self.start_timestamps]
        # end_timestamps = start + duration
        self.end_time = [start+timedelta(seconds=dur) for start, dur in zip(self.start_time, durations)]
        del audio_timestamp
        del csv_timestamp
        del common_timestamp
        del audio_files
        del durations

    def _read_single_file(self, file_timestamp, start, end):
        filename = file_timestamp + '.wav'
        ts_time = datetime.strptime(file_timestamp, self.timestamp_tmpl)

        offset = int((start - ts_time).total_seconds())
        total_duration = int((end - start).total_seconds())

        path = os.path.join(self.root, filename)
        y, sr = librosa.load(path)

        data_to_label = y[sr * offset:sr * int(offset + self.label_length)]
        timestamp_label = start.strftime(self.timestamp_tmpl)

        data_not_to_label = y[sr * int(offset + self.label_length):sr * int(offset + total_duration)]
        timestamp_not_to_label = (start + timedelta(seconds=self.label_length)).strftime(self.timestamp_tmpl)

        self._save_features(data_to_label, sr, self.label_path, timestamp_label)
        self._save_features(data_not_to_label, sr, self.unlabel_path, timestamp_not_to_label)

    def wrap_read_single_file(self, file_timestamp, start, end):
        try:
            self._read_single_file(file_timestamp, start, end)
        except Exception as e:
            path = os.path.join(self.root, file_timestamp + '.wav')
            self.logger.error("Failed to read file {:s} to load the data in range {:s} -> {:s}. Error message: {:s}".
                              format(path, start.strftime(self.timestamp_tmpl), end.strftime(self.timestamp_tmpl), str(e)))

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

    @staticmethod
    def _save_audio_file(audio, sample_rate, path, filename):
        save_path = os.path.join(path, filename+".wav")
        # in case of librosa.output deprecated or removed
        soundfile.write(save_path, audio, sample_rate)

    @staticmethod
    def _save_features(audio, sample_rate, path, filename):
        save_path = os.path.join(path, filename+".npy")
        features = librosa.feature.mfcc(y=audio, sr=sample_rate)
        np.save(save_path, features)

    def _get_duration(self, filename):
        wav, sr = librosa.load(os.path.join(self.root, filename))
        return librosa.get_duration(y=wav, sr=sr)
