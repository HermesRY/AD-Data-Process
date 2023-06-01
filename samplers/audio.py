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
    def __init__(self, root, num_workers=4, chunk_size=200,
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
        # self.target_path = target_path
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.label = sample_labeled_data
        self.unlabeled = sample_unlabeled_data
        self.save_raw = save_raw_data
        self.save_features = save_features
        assert label_rate <= sample_rate, 'label_rate should be less than sample_rate!'

        # make the needed directories
        self._check_paths()
        self._folder_navigation()

    def _folder_navigation(self):
        """
        Count the start/end time and the duration of each audio file under the root.
        """
        audio_files = [file for file in os.listdir(self.root) if file.endswith('.wav')]
        # calculate the total durations of the audio files under the root
        durations = [self._get_duration(file) for file in audio_files]
        self.filenames = audio_files
        self.durations = durations
        self.total_duration = sum(durations)
        # count the start timestamps
        self.start_timestamps = [os.path.splitext(file)[0] for file in audio_files]
        self.start_time = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in self.start_timestamps]
        # end_timestamps = start + duration
        self.end_time = [start+timedelta(seconds=dur) for start, dur in zip(self.start_time, self.durations)]
        self.end_timestamps = [t.strftime("%Y-%m-%d_%H-%M-%S") for t in self.end_time]
        del audio_files
        del durations

    def _check_paths(self):
        # path to the sampled data to be labeled
        self.label_raw_path = os.path.join(self.target_path, 'labeled', 'raw', 'audio')
        self.label_feature_path = os.path.join(self.target_path, 'labeled', 'features', 'audio')
        # path to the sampled data to be unlabeled
        self.unlabeled_raw_path = os.path.join(self.target_path, 'unlabeled', 'raw', 'audio')
        self.unlabeled_feature_path = os.path.join(self.target_path, 'unlabeled', 'features', 'audio')

        paths = [self.label_raw_path, self.label_feature_path,
                 self.unlabeled_feature_path, self.unlabeled_raw_path]

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

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

    def _sample_single_file(self, path):
        """
        To sample a single .wav audio file
        :param path: path to the .wav file
        """
        wav, sr = librosa.load(path)
        # get the timestamp from the filename
        start_timestamp = Path(path).stem
        start_time = datetime.strptime(start_timestamp, "%Y-%m-%d_%H-%M-%S")

        num_chunks = int(np.ceil(len(wav) / (self.chunk_size * sr)))
        # for each chunk, sample the data to be labeled and unlabeled.
        for i in range(num_chunks):
            # for the data to be labeled
            if self.label:
                start_idx = i * num_chunks * sr
                # end = start + sample_rate * chunk_size (10% * 200)
                end_idx = start_idx + self.label_rate * self.chunk_size * sr
                # avoid overflow
                end_idx = min(end_idx, len(wav))

                cur_time = start_time + timedelta(seconds=i * num_chunks)
                cur_timestamp = cur_time.strftime("%Y-%m-%d_%H-%M-%S")

                labeled_data = wav[start_idx:end_idx]
                if self.save_raw:
                    self._save_audio_file(labeled_data, sr, self.label_raw_path, cur_timestamp)
                if self.save_features:
                    self._save_features(labeled_data, sr, self.label_feature_path, cur_timestamp)

            # for data not to be labeled
            if self.unlabeled:
                start_idx = i * num_chunks * sr + self.label_rate * self.chunk_size * sr + 1
                end_idx = i * num_chunks * sr + self.sample_rate * self.chunk_size * sr
                end_idx = min(end_idx, len(wav))

                cur_time = start_time + timedelta(seconds=i * num_chunks + self.label_rate * self.chunk_size)
                cur_timestamp = cur_time.strftime("%Y-%m-%d_%H-%M-%S")

                unlabeled_data = wav[start_idx:end_idx]
                if self.save_raw:
                    self._save_audio_file(unlabeled_data, sr, self.unlabeled_raw_path, cur_timestamp)
                if self.save_features:
                    self._save_features(unlabeled_data, sr, self.unlabeled_feature_path, cur_timestamp)

    def sample(self):
        audio_files = [file for file in os.listdir(self.root) if file.endswith('.wav')]
        file_paths = [os.path.join(self.root, file) for file in audio_files]

        with Pool(self.num_workers) as pool:
            pool.map(self._sample_single_file, file_paths)
