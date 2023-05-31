import os
from datetime import datetime


class AlzheimerDataset:
    def __init__(self, root, start_time="7:00:00", end_time="19:00:00", num_workers=4):
        self.root = root
        self.start_time = start_time
        self.end_time = end_time
        self.num_workers = num_workers

        self.audio_root = os.path.join(self.root, 'audio')
        self.depth_root = os.path.join(self.root, 'depth')
        self.radar_root = os.path.join(self.root, 'radar')

    def _filter_hour_directories(self, path):
        """
        This function iterates all the directories under path
        and filter out those are between start_time and end_time
        """
        hour_strs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        start_time = datetime.strptime(self.start_time, '%H:%M:%S').time()
        end_time = datetime.strptime(self.end_time, '%H:%M:%S').time()
        filtered_hours = [d for d in hour_strs if start_time <= datetime.strptime(d, '%Y-%m-%d_%H-%M-%S').time() <= end_time]
        return filtered_hours

    def _check_common_hours(self):
        """
        Select the hours that all the three modalities have logged something.
        """
        audio_hours, depth_hours, radar_hours = self._filter_hour_directories(self.audio_root), \
                                                self._filter_hour_directories(self.depth_root),\
                                                self._filter_hour_directories(self.radar_root)

        audio_hours, depth_hours, radar_hours = set(audio_hours), set(depth_hours), set(radar_hours)
        common_hours = audio_hours.intersection(depth_hours, radar_hours)
        self.hours = list(common_hours)

