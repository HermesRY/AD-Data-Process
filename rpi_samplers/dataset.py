import os
import time
from datetime import datetime, timedelta
from multiprocessing import Process
from . import AudioSampler, DepthSampler, RadarSampler, Pool


class RpiAlzheimerDataset:
    def __init__(self, root, target_path, logger, chunk_size=200, sample_rate=.1,
                 label_rate=.01, start_time="7:00:00", end_time="19:00:00", num_workers=16, depth_only=False):
        self.root = root
        self.target_path = target_path
        self.logger = logger
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.start_time = start_time
        self.end_time = end_time
        self.num_workers = num_workers
        self.label_length = self.chunk_size * self.label_rate
        self.depth_only = depth_only

        # sensor status
        self.depth_sensor = False
        self.radar_sensor = False
        self.audio_sensor = False

        self.audio_root = os.path.join(self.root, 'audio')
        self.depth_root = os.path.join(self.root, 'depth')
        self.radar_root = os.path.join(self.root, 'radar')
        self._check_path()

    def _check_path(self):
        label_audio_path = os.path.join(self.target_path, 'label', 'audio')
        label_depth_path = os.path.join(self.target_path, 'label', 'depth')
        label_radar_path = os.path.join(self.target_path, 'label', 'radar')
        unlabel_audio_path = os.path.join(self.target_path, 'unlabel', 'audio')
        unlabel_depth_path = os.path.join(self.target_path, 'unlabel', 'depth')
        unlabel_radar_path = os.path.join(self.target_path, 'unlabel', 'radar')

        paths = [label_audio_path, label_depth_path, label_radar_path,
                 unlabel_audio_path, unlabel_depth_path, unlabel_radar_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
            del path

    def _init_sensors(self, audio_path, depth_path, radar_path, hour_datetime):
        audio = depth = radar = None
        if self.audio_sensor:
            audio = AudioSampler(audio_path, hour_datetime, self.target_path, self.logger, self.label_length)
        if self.depth_sensor:
            depth = DepthSampler(depth_path, self.target_path, self.logger, self.label_length)
        if self.radar_sensor:
            radar = RadarSampler(radar_path, hour_datetime, self.target_path, self.logger, self.label_length)
        return audio, depth, radar

    @staticmethod
    def _start_sample(audio, depth, radar, selected_times):
        p_list = []
        if audio is not None:
            p_list.append(Process(target=audio.sample, args=(selected_times,)))
        if depth is not None:
            p_list.append(Process(target=depth.sample, args=(selected_times,)))
        if radar is not None:
            p_list.append(Process(target=radar.sample, args=(selected_times,)))
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

    def _filter_hour_directories(self, timestamp):
        """
        This function iterates all the directories under path
        and select those are between start_time and end_time
        """
        try:
            start_time = datetime.strptime(self.start_time, '%H:%M:%S').time()
            end_time = datetime.strptime(self.end_time, '%H:%M:%S').time()
            filtered_hours = [d for d in timestamp
                              if start_time <= datetime.strptime(d, '%Y-%m-%d_%H-%M-%S').time() <= end_time]
        except FileNotFoundError:
            self.logger.warning("No audio sensor found in {:s}".format(self.root))
            filtered_hours = []
        return filtered_hours

    def _check_common_hours(self):
        """
        Select the hours that all the three sensors have logged something.
        """
        try:
            audio_strs = [d for d in os.listdir(self.audio_root) if d.endswith('.wav')]
            audio_strs = [item.split('_')[-1].split('.')[0] for item in audio_strs]
            audio_ts = [datetime.strptime(d, '%Y-%m-%d-%H-%M-%S') for d in audio_strs]
            audio_strs = [item.strftime('%Y-%m-%d_%H-00-00') for item in audio_ts]
        except FileNotFoundError:
            self.logger.warning("No audio sensor found in {:s}".format(self.root))
            audio_strs = []

        try:
            depth_strs = [d for d in os.listdir(self.depth_root) if os.path.isdir(os.path.join(self.depth_root, d))]
            depth_strs = [d for d in depth_strs if len(os.listdir(os.path.join(self.depth_root, d))) > 0]
        except FileNotFoundError:
            self.logger.warning("No depth sensor found in {:s}".format(self.root))
            depth_strs = []

        try:
            radar_strs = [d for d in os.listdir(self.radar_root) if d.endswith('.pkl')]
            radar_strs = [item.split('_')[-1].split('.')[0] for item in radar_strs]
            radar_ts = [datetime.strptime(d, '%Y-%m-%d-%H-%M-%S') for d in radar_strs]
            radar_strs = [item.strftime('%Y-%m-%d_%H-00-00') for item in radar_ts]
        except FileNotFoundError:
            self.logger.warning("No radar sensor found in {:s}".format(self.root))
            radar_strs = []

        audio_hours, depth_hours, radar_hours = self._filter_hour_directories(audio_strs), \
                                                self._filter_hour_directories(depth_strs), \
                                                self._filter_hour_directories(radar_strs)

        if len(depth_hours) >= 100:
            self.depth_sensor = True
        if len(radar_hours) >= 100:
            self.radar_sensor = True
        if len(audio_hours) >= 100:
            self.audio_sensor = True

        if self.depth_only:
            self.audio_sensor = False
            self.radar_sensor = False

        # if there is no audio or radar, use depth hours
        if self.depth_sensor:
            if not self.audio_sensor:
                audio_hours = depth_hours
            if not self.radar_sensor:
                radar_hours = depth_hours

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

        starts.sort()
        ends.sort()

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

    def check_single_hour_overlap(self, hour_ts):
        clock_start = time.time()
        hour_dt = datetime.strptime(hour_ts, '%Y-%m-%d_%H-%M-%S')
        depth_path = os.path.join(self.depth_root, hour_ts)

        with Pool(processes=3) as pool:
            audio, depth, radar = pool.starmap(self._init_sensors, [(self.audio_root, depth_path, self.radar_root, hour_dt)])[0]

        if self.depth_sensor:
            if self.audio_sensor and self.radar_sensor:
                start_time = audio.start_time + depth.start_time + radar.start_time
                end_time = audio.end_time + depth.end_time + radar.end_time
            elif not self.audio_sensor and not self.radar_sensor:
                self.logger.warning("No audio and radar sensor found in {:s}".format(self.root))
                start_time = depth.start_time + depth.start_time + depth.start_time
                end_time = depth.end_time + depth.end_time + depth.end_time
            elif not self.audio_sensor:
                self.logger.warning("No audio sensor found in {:s}".format(self.root))
                start_time = depth.start_time + depth.start_time + radar.start_time
                end_time = depth.end_time + depth.end_time + radar.end_time
            elif not self.radar_sensor:
                self.logger.warning("No radar sensor found in {:s}".format(self.root))
                start_time = audio.start_time + depth.start_time + depth.start_time
                end_time = audio.end_time + depth.end_time + depth.end_time
        else:
            self.logger.error("No depth sensor found in {:s}".format(self.root))
            return

        working_periods, working_time = self._find_all_working_regions(start_time, end_time)
        sample_size = self.chunk_size * self.sample_rate

        if working_time > timedelta(seconds=sample_size):
            self.logger.info("Find {:s} overlap in {:s} under {:s}".format(str(working_time), hour_ts, self.root))
            for start, end in working_periods:
                if (end - start) > timedelta(seconds=sample_size):
                    duration = int((end - start).total_seconds())
                    selected_times = [
                        (
                            start + timedelta(seconds=i * self.chunk_size),
                            min(start + timedelta(seconds=i * self.chunk_size + sample_size), end),
                        )
                        for i in range(duration // self.chunk_size + 2)
                        if start + timedelta(seconds=i * self.chunk_size) < end
                           and (
                                   duration % self.chunk_size >= sample_size
                                   or i < duration // self.chunk_size
                           )
                    ]
                    self._start_sample(audio, depth, radar, selected_times)

            self.logger.info("Finished sampling {:s} under {:s}. Time cost: {:f}"
                             .format(hour_ts, self.root, time.time() - clock_start))

        else:
            self.logger.warning("Overlap in {:s} under {:s} is {:s} less than {:.2f} seconds"
                                .format(hour_ts, self.root, str(working_time), sample_size))

    @staticmethod
    def _run_process_helper(func, names, num_workers):
        with Pool(num_workers) as pool:
            result = pool.map_async(func, names)

            # Wait for all processes to finish (ignoring the returned list of None values)
            result.get()

            # Close the pool and wait for all processes to complete
            pool.close()
            pool.join()

    def run(self):
        if not hasattr(self, 'hours'):
            self._check_common_hours()
        clock_start = time.time()

        self._run_process_helper(self.check_single_hour_overlap, self.hours, self.num_workers)
        time_cost = timedelta(seconds=time.time() - clock_start)
        self.logger.info(f"Finished sampling {self.root}. Total time cost: {time_cost}")
