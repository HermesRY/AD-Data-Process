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
    def __init__(self, root, target_path, logger, label_length, timestamp_tmpl="%Y-%m-%d_%H-%M-%S", num_workers=4, chunk_size=200,
                 sample_rate=.1, label_rate=.01, sample_labeled_data=True,
                 sample_unlabeled_data=True, save_raw_data=True, save_features=True):
        """
        This class samples a specified percentage of data from audio files under a specified folder,
        and saves the sampled data to a target path. The audio files are separated into chunks whose size
        is specified by the `chunk_size` parameter (default: 200 seconds).

        :param root: path of the audio files, generated every hour, e.g., ./data/audio/2022-11-21_10-00-00/
        :param target_path: path to store the sampled data
        :param num_workers: number of workers
        :param chunk_size: The size of the basic chunk (default: 200 seconds).
        :param sample_rate: The proportion of data to be sampled (default: 10%).
                            The first `sample_rate` percentage of every chunk is to be sampled.
        :param label_rate: The proportion of data to be labeled (default: 1%).
                            The first `label_rate` percentage of each chunk is to be labeled.
        :param sample_labeled_data: If True, data to be labeled will be sampled.
        :param sample_unlabeled_data: If True, data not to be labeled will be sampled.
        :param save_raw_data: If True, the raw sampled data (i.e., the wav file) will be saved.
        :param save_features: If True, the processed data (i.e., the features) will be saved.

        """
        self.root = root
        self.target_path = target_path
        self.logger = logger
        self.label_length = label_length
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.label = sample_labeled_data
        self.unlabeled = sample_unlabeled_data
        self.save_raw = save_raw_data
        self.save_features = save_features
        self.timestamp_tmpl = timestamp_tmpl
        assert label_rate <= sample_rate, 'label_rate should be less than sample_rate!'

        # make the needed directories
        self._folder_navigation()

    def _folder_navigation(self):
        """
        Count the start/end time and the duration of each audio file under the root.
        """
        audio_files = [file for file in os.listdir(self.root) if file.endswith('.wav')]
        durations = [self._get_duration(file) for file in audio_files]
        # calculate the total durations of the audio files under the root
        self.filenames = audio_files
        # count the start timestamps
        self.start_timestamps = [os.path.splitext(file)[0] for file in audio_files]
        self.start_time = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in self.start_timestamps]
        # end_timestamps = start + duration
        self.end_time = [start+timedelta(seconds=dur) for start, dur in zip(self.start_time, durations)]
        del audio_files
        del durations

    def _read_single_file(self, file_timestamp, start, end):
        filename = file_timestamp + '.wav'
        ts_time = datetime.strptime(file_timestamp, self.timestamp_tmpl)

        offset = (start - ts_time).total_seconds()
        total_duration = (end - start).total_seconds()

        path = os.path.join(self.root, filename)
        y, sr = librosa.load(path)

        data_to_label = y[sr * offset:sr * (offset + self.label_length)]
        data_not_to_label = y[sr * (offset + self.label_length):sr * (offset + total_duration)]

        self.logger.info("audio shape(label): {:s}".format(str(data_to_label.shape)))
        self.logger.info("audio shape(unlabeled): {:s}".format(str(data_not_to_label.shape)))
        # save the data as mfcc

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
                    pool.apply_async(self._read_single_file, args=(file_timestamp, start, end))
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
