import os
from datetime import datetime, timedelta
from multiprocessing import Pool
from samplers import AudioSampler, DepthSampler, RadarSampler


class AlzheimerDataset:
    def __init__(self, root, start_time="7:00:00", end_time="19:00:00", num_workers=4):
        self.root = root
        self.start_time = start_time
        self.end_time = end_time
        self.num_workers = num_workers

        self.audio_root = os.path.join(self.root, 'audio')
        self.depth_root = os.path.join(self.root, 'depth')
        self.radar_root = os.path.join(self.root, 'radar')

    @staticmethod
    def _init_sensors(audio_path, depth_path, radar_path):
        audio = AudioSampler(audio_path)
        depth = DepthSampler(depth_path)
        radar = RadarSampler(radar_path)
        return audio, depth, radar

    def _filter_hour_directories(self, path):
        """
        This function iterates all the directories under path
        and select those are between start_time and end_time
        """
        hour_strs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # filter out empty folders
        hour_strs = [d for d in hour_strs if len(os.listdir(os.path.join(path, d))) > 0]
        start_time = datetime.strptime(self.start_time, '%H:%M:%S').time()
        end_time = datetime.strptime(self.end_time, '%H:%M:%S').time()
        filtered_hours = [d for d in hour_strs if
                          start_time <= datetime.strptime(d, '%Y-%m-%d_%H-%M-%S').time() <= end_time]
        return filtered_hours

    def _check_common_hours(self):
        """
        Select the hours that all the three sensors have logged something.
        """
        audio_hours, depth_hours, radar_hours = self._filter_hour_directories(self.audio_root), \
                                                self._filter_hour_directories(self.depth_root), \
                                                self._filter_hour_directories(self.radar_root)

        audio_hours, depth_hours, radar_hours = set(audio_hours), set(depth_hours), set(radar_hours)
        common_hours = audio_hours.intersection(depth_hours, radar_hours)
        self.hours = list(common_hours)

    @staticmethod
    def _find_all_working_regions(starts, ends):
        """
        Find all the time regions that all the three sensors start work and none of them ends.
        :param starts: a list of datetime, indicating the start time points for all the three sensors
        :param ends: a list of datetime, indicating the end time points for all the three sensors
        :return: all_working_regions: a list of tuple (start, end)
        :return total_working_time: total time length of all working in seconds
        """
        all_working_regions = []
        total_working_time = timedelta()
        start_idx = end_idx = working_sensors = 0
        current_start = None

        while start_idx < len(starts) and end_idx < len(ends):
            if starts[start_idx] < ends[end_idx]:
                working_sensors += 1
                if working_sensors == 3:
                    current_start = starts[start_idx]
                start_idx += 1
            else:
                working_sensors -= 1
                if working_sensors == 2 and current_start is not None:
                    all_working_regions.append((current_start, ends[end_idx]))
                    current_start = None
                end_idx += 1

        for start, end in all_working_regions:
            total_working_time += end - start

        return all_working_regions, total_working_time

    def check_single_hour_overlap(self, folder):
        audio_path, depth_path, radar_path = os.path.join(self.audio_root, folder),\
                                             os.path.join(self.depth_root, folder), \
                                             os.path.join(self.radar_root, folder)

        with Pool(processes=3) as pool:
            audio, depth, radar = pool.starmap(self._init_sensors, [(audio_path, depth_path, radar_path)])[0]

        start_time = [audio.start_time, depth.start_time, radar.start_time]
        end_time = [audio.end_time, depth.end_time, radar.end_time]

        working_periods, working_time = self._find_all_working_regions(start_time, end_time)
        del audio
        del depth
        del radar

        return working_periods, working_time




