import math
import argparse
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import librosa
import cv2
import pandas as pd
from multiprocessing import Pool
import copy
import shutil


def preprocess_audio(audio_path, start_time, end_time, npy_save_path, args):
    """
    This function is used before training, it transforms raw audio to .npy file.
    Including slice and MFCC generation
    """
    if not os.path.exists(audio_path):
        print("[Preprocess] Error: Cannot find directory: ", audio_path)
        return
    audio_paths = []
    audio_folders = os.listdir(audio_path)
    for audio_folder in audio_folders:
        if 'txt' in audio_folder:
            continue
        audio_files = os.listdir(os.path.join(audio_path, audio_folder))
        for audio_file in audio_files:
            audio_paths.append(os.path.join(audio_folder, audio_file))
    list.sort(audio_paths, key=lambda x: datetime.strptime(os.path.split(x)[1][:-4], "%Y-%m-%d_%H-%M-%S"))
    start_flag = False
    for audio_file in audio_paths:
        audio_time = datetime.strptime(os.path.split(audio_file)[1][:-4], "%Y-%m-%d_%H-%M-%S")
        # If we require data from 12:00:00, audio 11:58:10.wav will be used
        if audio_time < start_time - timedelta(minutes=10):
            continue
        # If end time is 13:00:00, audio 12:58:10 will be used
        if not start_flag:
            wav, sr = librosa.load(os.path.join(audio_path, audio_file))
            unused_duration = start_time - audio_time
            unused_duration = unused_duration.seconds
            if audio_time + timedelta(minutes=10) > end_time:
                used_duration = end_time - audio_time
                used_duration = used_duration.seconds
                result_wav = wav[unused_duration * sr:used_duration * sr]
                break
            else:
                result_wav = wav[unused_duration * sr:]
            # init the first audio
            start_flag = True
            continue
        if audio_time > end_time - timedelta(minutes=10):
            wav, sr = librosa.load(os.path.join(audio_path, audio_file))
            used_duration = end_time - audio_time
            used_duration = used_duration.seconds
            # combine the last audio and exit loop
            result_wav = np.hstack((result_wav, wav[:used_duration * sr]))
            break

        # integer_time for 12:58:10.wav is 12:50:00. If we require data before 13:00:00, this audio will be used
        wav, sr = librosa.load(os.path.join(audio_path, audio_file))
        result_wav = np.hstack((result_wav, wav))
    del audio_files

    wav_length = int(librosa.get_duration(y=result_wav, sr=sr))
    if wav_length < 60:
        print(f"[Preprocess] {wav_length} seconds is not enough. Part id = {args.part_id}.")
        return False

    wav_length = wav_length - wav_length % args.clip_duration
    #print(f"[Preprocess] Finish audio load: {wav_length} seconds, start to process audio.")
    mfcc_matrix = []
    for i in range(0, wav_length, args.clip_duration):
        slice_wav = result_wav[i * sr: (i + args.reserved_time) * sr]
        slice_mfcc = librosa.feature.mfcc(y=slice_wav, sr=sr)
        mfcc_matrix.append(slice_mfcc)

    mfcc_matrix = np.array(mfcc_matrix)
    # 如果sample比正常小太多，说明audio miss严重，那么就舍弃整个半小时的数据
    # while mfcc_matrix.shape[0] < 3600:
    #     mfcc_matrix = np.concatenate((mfcc_matrix, np.zeros((1, 20, 87))), axis=0)
    np.save(npy_save_path, mfcc_matrix)
    #print(f"[Preprocess] Finish audio process {mfcc_matrix.shape} and saved in {npy_save_path}.")
    return True


def preprocess_depth(depth_path, start_time, end_time, npy_save_path, args):
    if not os.path.exists(depth_path):
        print("[Preprocess] Error: This depth data is not existed: ", depth_path)
        return None

    depth_folders = os.listdir(depth_path)
    list.sort(depth_folders, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))
    start_flag = False
    stop_flag = False
    total_duration = end_time - start_time
    total_duration = total_duration.seconds
    target_frame_num = int(total_duration / args.clip_duration * args.reserved_time * 15)
    assert target_frame_num >= 30, "Total duration is too short!!!"

    result_video = np.empty((target_frame_num, 480, 640))
    _cur_frame = 0

    frame_pos = 0           # target pos of frame that stores in array
    clip_frame_num = 0      # less than args.reserved_time * 15
    record_flag = False     # if true, clip is recorded in array. Or pass it.

    for folder in depth_folders:
        # the folder time is always integer. We concern start time is 11:50:00, folder 11:00:00 will be used.
        # Also, if start_time is 12:01:00, this video may be in folder 11:00:00
        if stop_flag:
            break
        if datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S") < start_time - timedelta(hours=1):
            continue
        videos = os.listdir(os.path.join(depth_path, folder))
        try:
            videos.remove('labels')
        except:
            pass
        for video in videos:
            if video[0] == '.':
                videos.remove(video)
        list.sort(videos, key=lambda x: datetime.strptime(x[:-4], "%Y-%m-%d_%H-%M-%S"))
        label_path = os.path.join(os.path.join(depth_path, folder, 'labels'))
        for video in videos:
            if os.path.splitext(video)[-1] == ".csv":
                continue
            video_time = datetime.strptime(video[:-4], "%Y-%m-%d_%H-%M-%S")
            video_path = os.path.join(depth_path, folder, video)
            # find the start time
            if not start_flag:
                if video_time <= start_time <= video_time + timedelta(minutes=10):
                    # find the split time in first video
                    cap = cv2.VideoCapture(video_path)
                    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_frame2 = total_frame
                    if total_frame == 0:
                        print(f"[Preprocess] Error: Cannot read: {video_path}")
                        return
                    csv_path = video_path[:-3] + 'csv'
                    csv_data = pd.read_csv(csv_path, header=None).dropna(axis="rows", how="all").values
                    start_frame = 1
                    split_frame = -1
                    for row in csv_data[1:]:
                        if datetime.strptime(row[0], "%Y-%m-%d_%H-%M-%S.%f") >= start_time:
                            split_frame = int(row[1])
                            break
                    for row in csv_data[1:]:
                        if datetime.strptime(row[0], "%Y-%m-%d_%H-%M-%S.%f") > end_time:
                            total_frame2 = int(row[1])
                            break
                    del csv_data
                    total_frame = min(total_frame, total_frame2)
                    if split_frame == -1:
                        print(f"[Preprocess] Error: Cannot find time {start_frame} in {csv_path}.")
                    start_flag = True

                    # create the first video
                    current_frame = start_frame
                    while 1:
                        ret = cap.grab()
                        if ret:
                            current_frame += 1
                            if current_frame < split_frame:
                                pass
                            else:
                                if clip_frame_num >= args.reserved_time * 15:
                                    # stop recording
                                    record_flag = False
                                    clip_frame_num = 0

                                if _cur_frame % (args.clip_duration * 15) == 0:
                                    record_flag = True

                                if record_flag:
                                    ret, frame = cap.retrieve()
                                    frame = np.mean(frame, axis=2)#[:, 80:560]
                                    #frame = cv2.resize(frame, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
                                    frame = np.expand_dims(frame, axis=0)
                                    if frame_pos >= target_frame_num:
                                        break
                                    result_video[frame_pos] = frame
                                    frame_pos += 1
                                    clip_frame_num += 1
                                _cur_frame += 1
                        else:
                            print(f"[Preprocess] Error: Cannot read frame {video_path}.")
                        if current_frame >= total_frame:
                            break
                else:
                    continue
            else:
                if end_time > video_time + timedelta(minutes=10):
                    cap = cv2.VideoCapture(video_path)
                    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    current_frame = 0
                    if total_frame == 0:
                        print(f"[Preprocess] Error: Cannot read: {video_path}")
                        return
                    # Common condition

                    while 1:
                        ret = cap.grab()
                        current_frame += 1
                        if ret:
                            if clip_frame_num >= args.reserved_time * 15:
                                # stop recording
                                record_flag = False
                                clip_frame_num = 0

                            if _cur_frame % (args.clip_duration * 15) == 0:
                                record_flag = True

                            if record_flag:
                                ret, frame = cap.retrieve()
                                frame = np.mean(frame, axis=2)#[:, 80:560]
                                #frame = cv2.resize(frame, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
                                frame = np.expand_dims(frame, axis=0)
                                result_video[frame_pos] = frame
                                clip_frame_num += 1
                                frame_pos += 1 
                            _cur_frame += 1
                        else:
                            print(f"[Preprocess] Error: Cannot read frame {video_path}.")
                        if current_frame >= total_frame:
                            break
                elif video_time <= end_time < video_time + timedelta(minutes=10):
                    # find the split time in last video
                    cap = cv2.VideoCapture(video_path)
                    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frame == 0:
                        print(f"[Preprocess] Error: Cannot read: {video_path}")
                        return
                    csv_path = video_path[:-3] + 'csv'
                    csv_data = pd.read_csv(csv_path, header=None).dropna(axis="rows", how="all").values
                    start_frame = 1
                    split_frame = -1
                    for row in csv_data[1:]:
                        if datetime.strptime(row[0], "%Y-%m-%d_%H-%M-%S.%f") >= end_time:
                            split_frame = int(row[1])
                            break
                    del csv_data
                    if split_frame == -1:
                        print(f"[Preprocess] Error: Cannot find time {start_frame} in {csv_path}.")

                    # create the first video
                    current_frame = start_frame
                    while 1:
                        ret = cap.grab()
                        if ret:
                            current_frame += 1
                            if clip_frame_num >= args.reserved_time * 15:
                                # stop recording
                                record_flag = False
                                clip_frame_num = 0

                            if _cur_frame % (args.clip_duration * 15) == 0:
                                record_flag = True

                            if record_flag:
                                ret, frame = cap.retrieve()
                                frame = np.mean(frame, axis=2)#[:, 80:560]
                                #frame = cv2.resize(frame, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
                                frame = np.expand_dims(frame, axis=0)
                                try:
                                    result_video[frame_pos] = frame
                                except:
                                    break
                                clip_frame_num += 1
                                frame_pos += 1 
                            _cur_frame += 1
                        else:
                            print(f"[Preprocess] Error: Cannot read frame {video_path}.")
                        if current_frame >= total_frame:
                            break
                    stop_flag = True
                    break  # quit out loop
    # result_video = result_video[:_cur_frame]
    if frame_pos < 90:
        return False
    #result_video = __reshape_depth(result_video)
    result_video = result_video.reshape((-1, 30, 480, 640))
    np.save(npy_save_path, result_video[:target_frame_num])
    del result_video
    return True


def __reshape_depth(result_video):
    # resample the raw video: 30 -> 16
    total_frame = result_video.shape[0]
    target_frame_list = np.array([*range(0, total_frame, 2)])
    target_frame_list2 = np.array([*range(0, total_frame, 15)])
    target_frame_list = np.unique(np.hstack((target_frame_list, target_frame_list2)))
    result_video = result_video[target_frame_list]
    while 1:
        try:
            result_video = result_video.reshape((-1, 16, 480, 640))
            break
        except:
            result_video = np.concatenate((result_video, np.zeros((1, 480, 640))), axis=0)
    return result_video


def preprocess_radar(radar_path, start_time, end_time, npy_save_path, args):
    if not os.path.exists(radar_path):
        print("[Preprocess] Error: This radar data is not existed: ", radar_path)
        return False
    radar_paths = []
    radar_folders = os.listdir(radar_path)
    for radar_folder in radar_folders:
        if 'txt' in radar_folder:
            continue
        radar_files = os.listdir(os.path.join(radar_path, radar_folder))
        for radar_file in radar_files:
            radar_paths.append(os.path.join(radar_folder, radar_file))
    list.sort(radar_paths, key=lambda x: datetime.strptime(os.path.split(x)[1][:-4], "%Y-%m-%d_%H-%M-%S"))
    radar_data = []

    sample_start_time = start_time
    sample_end_time = start_time + timedelta(seconds=args.reserved_time)

    record_flag = False
    total_frame_num = int((end_time-start_time).total_seconds()) / args.clip_duration * args.reserved_time * 10

    for radar_file in radar_paths:
        radar_time = datetime.strptime(os.path.split(radar_file)[1][:-4], "%Y-%m-%d_%H-%M-%S")
        # find the first file
        if radar_time < start_time - timedelta(minutes=15, seconds=2):
            continue
        if radar_time > end_time:
            break

        while sample_start_time < radar_time:
            record_flag = False
            sample_start_time += timedelta(seconds=args.clip_duration)
            sample_end_time = sample_start_time + timedelta(seconds=args.reserved_time)
        if radar_time <= start_time <= radar_time + timedelta(minutes=15, seconds=2):
            # first file
            file_path = os.path.join(radar_path, radar_file)
            temp_data = np.load(file_path, allow_pickle=True)
            for frame in temp_data:
                frame_time = datetime.strptime(frame['Time'], '%Y%m%d-%H%M%S-%f')
                if frame_time <= end_time:
                    if frame_time >= start_time:
                        # first frame
                        if not record_flag:
                            if sample_start_time < frame_time < sample_end_time:
                                record_flag = True
                        if record_flag:
                            if frame_time > sample_end_time:
                                record_flag = False
                                sample_start_time += timedelta(seconds=args.clip_duration)
                                sample_end_time = sample_start_time + timedelta(seconds=args.reserved_time)
                            else:
                                radar_data.append(__reshape_radar(frame['Data']))
                else:
                    continue
        # find the last file
        elif radar_time <= end_time <= radar_time + timedelta(minutes=15, seconds=2):
            file_path = os.path.join(radar_path, radar_file)
            temp_data = np.load(file_path, allow_pickle=True)
            first_frame_time = datetime.strptime(temp_data[0]['Time'], '%Y%m%d-%H%M%S-%f')

            for frame in temp_data:
                frame_time = datetime.strptime(frame['Time'], '%Y%m%d-%H%M%S-%f')
                if frame_time <= end_time:
                    if not record_flag:
                        if sample_start_time < frame_time < sample_end_time:
                            record_flag = True
                    if record_flag:
                        if frame_time > sample_end_time:
                            record_flag = False
                            sample_start_time += timedelta(seconds=args.clip_duration)
                            sample_end_time = sample_start_time + timedelta(seconds=args.reserved_time)
                        else:
                            radar_data.append(__reshape_radar(frame['Data']))
                else:
                    continue

        # common condition
        else:
            file_path = os.path.join(radar_path, radar_file)
            temp_data = np.load(file_path, allow_pickle=True)
            first_frame_time = datetime.strptime(temp_data[0]['Time'], '%Y%m%d-%H%M%S-%f')

            for frame in temp_data:
                frame_time = datetime.strptime(frame['Time'], '%Y%m%d-%H%M%S-%f')
                if frame_time <= end_time:
                    if not record_flag:
                        if sample_start_time < frame_time < sample_end_time:
                            record_flag = True
                    if record_flag:
                        if frame_time > sample_end_time:
                            record_flag = False
                            sample_start_time += timedelta(seconds=args.clip_duration)
                            sample_end_time = sample_start_time + timedelta(seconds=args.reserved_time)
                        else:
                            radar_data.append(__reshape_radar(frame['Data']))
                else:
                    continue
        # after processing each file, tackle the radar down condition
        #if record_flag:
        #    record_flag = False
        #    current_frame_num = len(radar_data)
        #    remove_frame_num = current_frame_num % (args.reserved_time * 10)
        #    for _ in range(remove_frame_num):

    radar_data = np.array(radar_data)
    if radar_data.shape[0] < 60:
        return False
    while 1:
        try:
            radar_data = radar_data.reshape((-1, 20, 2, 16, 32, 16))
            break
        except:
            radar_data = np.concatenate((radar_data, np.zeros((1,2,16,32,16))), axis=0)
    #while radar_data.shape[0] < 3600:
    #    radar_data = np.concatenate((radar_data, np.zeros((1,20,2,16,32,16))), axis=0)
    np.save(npy_save_path, radar_data)
    return True


def __reshape_radar(data):
    """
    Reshape one-frame radar data to (2,16,32,16)
    """
    voxel_occ = np.zeros((16, 32, 16))
    voxel_vel = np.zeros((16, 32, 16))
    num_points = data.shape[1]
    for point_idx in range(num_points):
        x, y, z, v = data[:, point_idx]
        x_loc = int(np.clip((x + 8) // 0.5, 0, 15))
        y_loc = int(np.clip(y // 0.25, 0, 31))  # int(y//0.25)
        z_loc = int(np.clip((z + 8) // 0.5, 0, 15))
        v = np.clip(v, -4, 4)

        # mapping in human tracking:
        # x: (-8, 8), 16 grid, res=1m
        # y: (0, 8), 32 grid, res=0.25m
        # z: (-8, 8), 16 grid, res=1m
        # v: (-4, 4)

        # print(frame_idx, x_loc, y_loc, z_loc)
        voxel_occ[x_loc, y_loc, z_loc] += 1
        voxel_vel[x_loc, y_loc, z_loc] += v

    voxel_vel /= (voxel_occ + 1e-6)  # mean velocity

    # print(voxel_vel.shape, voxel_occ.shape, clip)   # (16, 32, 16)
    voxel = np.stack((voxel_occ, voxel_vel), axis=0)  # (2, 16, 32, 16)
    return voxel


def drop_data(depth_path, audio_path, radar_path):
    # delete data in all sensors that missed in radar_path
    if os.path.exists(depth_path):
        depth = np.load(depth_path)
        split_save(depth_path, depth)
        os.remove(depth_path)
        del depth

    if os.path.exists(audio_path):
        audio = np.load(audio_path)
        split_save(audio_path, audio)
        os.remove(audio_path)
        del audio

    if os.path.exists(radar_path):
        radar = np.load(radar_path)
        split_save(radar_path, radar)
        os.remove(radar_path)
        del radar


def split_save(save_path, data):
    total_frame = data.shape[0]
    loops = math.floor(total_frame)

    for i in range(0, loops):
        file_save_path = save_path[:-4]+'_'+str(i)+'.npy'
        np.save(file_save_path, data[i])


def combine_all(temp_args, total_num):
    start_index = 0
    radar_save_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), "radar")
    audio_save_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), "audio")
    depth_save_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), "depth")
    if not os.path.exists(radar_save_path):
        os.mkdir(radar_save_path)
    if not os.path.exists(audio_save_path):
        os.mkdir(audio_save_path)
    if not os.path.exists(depth_save_path):
        os.mkdir(depth_save_path)
    for part_id in range(total_num):
        radar_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), str(part_id), "radar")
        audio_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), str(part_id), "audio")
        depth_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), str(part_id), "depth")
        radar_list = os.listdir(radar_path)
        audio_list = os.listdir(audio_path)
        depth_list = os.listdir(depth_path)

        if len(radar_list) == len(audio_list) and len(audio_list) == len(depth_list):
            file_num = len(radar_list)
            for idx in range(file_num):
                shutil.move(os.path.join(radar_path, f"radar_{idx}.npy"), os.path.join(radar_save_path, f"radar_{idx+start_index}.npy"))
                shutil.move(os.path.join(audio_path, f"audio_{idx}.npy"), os.path.join(audio_save_path, f"audio_{idx+start_index}.npy"))
                shutil.move(os.path.join(depth_path, f"depth_{idx}.npy"), os.path.join(depth_save_path, f"depth_{idx+start_index}.npy"))
            start_index += file_num
        else:
            print(f"{part_id} has different files amount. Delete")
            #print(len(radar_list))
            #print(len(audio_list))
            #print(len(depth_list))
        shutil.rmtree(os.path.join(temp_args.save_data_root, str(temp_args.subject_id), str(part_id)))


def main_thread(temp_args):
    radar_path = os.path.join(temp_args.raw_data_root, str(temp_args.subject_id), "data", "radar")
    audio_path = os.path.join(temp_args.raw_data_root, str(temp_args.subject_id), "data", "audio")
    depth_path = os.path.join(temp_args.raw_data_root, str(temp_args.subject_id), "data", "depth")

    radar_save_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), str(temp_args.part_id), "radar", f"radar.npy")
    audio_save_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), str(temp_args.part_id), "audio", f"audio.npy")
    depth_save_path = os.path.join(temp_args.save_data_root, str(temp_args.subject_id), str(temp_args.part_id), "depth", f"depth.npy")
    
    if not os.path.exists(os.path.split(radar_save_path)[0]):
        os.makedirs(os.path.split(radar_save_path)[0])
    if not os.path.exists(os.path.split(audio_save_path)[0]):
        os.makedirs(os.path.split(audio_save_path)[0])
    if not os.path.exists(os.path.split(depth_save_path)[0]):
        os.makedirs(os.path.split(depth_save_path)[0])

    start_time = datetime.strptime(temp_args.start_time, "%Y-%m-%d-%H-%M-%S")
    end_time = datetime.strptime(temp_args.end_time, "%Y-%m-%d-%H-%M-%S")

    if temp_args.calibration:
        time_diff = datetime.strptime(temp_args.cali_time_depth, "%Y-%m-%d-%H-%M-%S") - datetime.strptime(temp_args.cali_time_sensor, "%Y-%m-%d-%H-%M-%S")
        sensor_start_time = start_time - time_diff
        sensor_end_time = end_time - time_diff
    else:
        sensor_start_time = start_time
        sensor_end_time = end_time
    ret = preprocess_audio(audio_path, sensor_start_time, sensor_end_time, audio_save_path, temp_args)
    if not ret:
        print(f"{temp_args.part_id}: audio load failed")
        return
    ret = preprocess_depth(depth_path, start_time, end_time, depth_save_path, temp_args)
    if not ret:
        print(f"{temp_args.part_id}: depth load failed")
        return
    ret = preprocess_radar(radar_path, sensor_start_time, sensor_end_time, radar_save_path, temp_args)
    if not ret:
        print(f"{temp_args.part_id}: radar load failed")
        return
    # print(f"{temp_args.part_id}: radar complete. Miss sample idx is: {miss_data[0]}")
    drop_data(depth_save_path, audio_save_path, radar_save_path)
    print(f"{temp_args.part_id} finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # You must set these arguments
    parser.add_argument('--start_time', type=str, help='start time of depth data')
    parser.add_argument('--end_time', type=str, help='end time of depth data')
    parser.add_argument('--calibration', action="store_true", help='use it when depth time cannot align with '
                                                                   'multi-sensor time')
    parser.add_argument('--cali_time_depth', type=str, help='if calibration == True, set depth\'s first start time;\n'
                                                            'Attention: it\'s different from start time, it is the '
                                                            'recorded start time of box in depth')
    parser.add_argument('--cali_time_sensor', type=str, help='if calibration == True, set multi-sensor\'s first start '
                                                             'time;\nAttention: it\'s different from start time, '
                                                             'it is the recorded start time of box in multi-sensor')
    parser.add_argument('--clip_duration', type=int, default=20, help='length of one clip before dropping (seconds)')
    parser.add_argument('--reserved_time', type=int, default=2, help='length of one clip after dropping (seconds)')
    parser.add_argument('--subject_id', type=str, help='Subject id should be like V00x')
    parser.add_argument('--part_id', type=int, default=0, help='one subject\'s data will be divided into several '
                                                               'parts for multiprocess')

    # You also must set these arguments
    parser.add_argument('--raw_data_root', type=str, default='/mnt/nas', help='root path of raw data')
    parser.add_argument('--save_data_root', type=str, default='complete_data', help='root path of processed data')
    parser.add_argument('--total_part_num', type=int, help='total time / 1 minutes')
    args = parser.parse_args()
    arg_list = []

    start_time = datetime.strptime(args.start_time, "%Y-%m-%d-%H-%M-%S")
    end_time = start_time + timedelta(minutes=1)
    temp_arg = copy.deepcopy(args)
    arg_list.append(temp_arg)
    idx = 1
    for i in range(1, args.total_part_num):
        start_time = start_time + timedelta(minutes=1)
        end_time = start_time + timedelta(minutes=1)
        if start_time.hour < 7 or start_time.hour >= 16:
            continue
        temp_arg = copy.deepcopy(args)
        temp_arg.start_time = datetime.strftime(start_time, "%Y-%m-%d-%H-%M-%S")
        temp_arg.end_time = datetime.strftime(end_time, "%Y-%m-%d-%H-%M-%S")
        temp_arg.part_id = idx
        idx += 1
        arg_list.append(temp_arg)
    
    p = Pool(32)
    for arg in arg_list:
        print(f"{arg.part_id} start: {arg.start_time}")
        p.apply_async(main_thread, args=(arg,))
    p.close()
    p.join()
    combine_all(args, idx)
    print("Finish all process!")
