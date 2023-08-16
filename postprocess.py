import os
import cv2
import numpy as np
import pandas as pd
from nx_samplers import Pool
from datetime import datetime
from multiprocessing import Process


data_path = "/pm1733_x3/sample_li"
filter_path = "/pm1733_x3/filtered_videos"
target_path = "/pm1733_x3/features_li"
depth_frames = 16
depth_shape = (16, 112, 112)
radar_shape = (20, 2, 16, 32, 16)
audio_shape = (20, 87)
max_workers = 8

def process_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    # iterate through the frames
    while len(frames) < depth_frames:
        # read the next frame
        ret, frame = cap.read()

        # if there are no more frames, break out of the loop
        if not ret:
            break

        # convert the frame to gray-scale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # add the gray-scale frame to the list
        frames.append(frame_gray)

    # if there are less than max_frames frames, interpolate to compensate
    while len(frames) < depth_frames:
        # get the last frame in the list
        last_frame = frames[-1]

        # interpolate to generate a new frame
        new_frame = last_frame

        # add the new frame to the list
        frames.append(new_frame)

    # release the video capture object
    cap.release()
    # to features
    features = np.array(frames)
    return features

def process_radar(radar, target_shape):
    if radar.shape == target_shape:
        return radar
    elif radar.shape[0] > target_shape[0]:
        return radar[:target_shape[0]]
    else:
        res = target_shape[0] - radar.shape[0]
        return np.concatenate((radar, radar[:res]), axis=0)

def _make_modality_path(root):
    modal = ['audio', 'depth', 'radar']
    for m in modal:
        if not os.path.exists(os.path.join(root, m)):
            os.makedirs(os.path.join(root, m))


def _process_single_time(source_root, target_root, timestamp, id):
    audio_path = os.path.join(source_root, 'audio')
    depth_path = os.path.join(source_root, 'depth')
    radar_path = os.path.join(source_root, 'radar')
    try:
        audio_file = np.load(os.path.join(audio_path, timestamp+'.npy'))
    except Exception as e:
        audio_file = None
    try:
        radar_file = np.load(os.path.join(radar_path, timestamp+'.npy'))
        radar_file = process_radar(radar_file, radar_shape)
    except Exception as e:
        radar_file = None
    try:
        depth_features = process_video(os.path.join(depth_path, timestamp+'.mp4'))
    except Exception as e:
        depth_features = None
        print(f"Failed to load {timestamp} depth data under {source_root}. Error message {e}")
    try:
        if depth_features is not None:
            if audio_file is not None and radar_file is not None:
                if audio_file.shape == audio_shape and radar_file.shape == radar_shape and depth_features.shape == depth_shape:
                    np.save(os.path.join(target_root, 'audio', str(id)+'.npy'), audio_file)
                    np.save(os.path.join(target_root, 'depth', str(id)+'.npy'), depth_features)
                    np.save(os.path.join(target_root, 'radar', str(id)+'.npy'), radar_file)
                else:
                    print(f"Shape not equal! Depth {depth_features.shape}; Radar {radar_file.shape}; Audio: {audio_file.shape}")
            elif audio_file is not None:
                if audio_file.shape == audio_shape and depth_features.shape == depth_shape:
                    np.save(os.path.join(target_root, 'audio', str(id)+'.npy'), audio_file)
                    np.save(os.path.join(target_root, 'depth', str(id)+'.npy'), depth_features)
                else:
                    print(f"Shape not equal! Depth {depth_features.shape}; Audio {audio_file.shape}")
            elif radar_file is not None:
                if radar_file.shape == radar_shape and depth_features.shape == depth_shape:
                    np.save(os.path.join(target_root, 'depth', str(id)+'.npy'), depth_features)
                    np.save(os.path.join(target_root, 'radar', str(id)+'.npy'), radar_file)
                else:
                    print(f"Shape not equal! Depth {depth_features.shape}; Radar {radar_file.shape}")
            else:
                if depth_features.shape == depth_shape:
                    np.save(os.path.join(target_root, 'depth', str(id)+'.npy'), depth_features)
                else:
                    print(f"Shape not equal! Depth {depth_features.shape}")
    except Exception as e:
        print(f"Failed to process {timestamp} data under {source_root}. Error message {e}")


def _process_single_subject(cur_root, target_root, yolo=None, workers=8):
    _make_modality_path(target_root)
    cur_depth_ts = [item.split('.')[0] for item in os.listdir(os.path.join(cur_root, 'depth')) if item.endswith('.mp4')]
    cur_radar_ts = [item.split('.')[0] for item in os.listdir(os.path.join(cur_root, 'radar')) if item.endswith('.npy')]
    cur_audio_ts = [item.split('.')[0] for item in os.listdir(os.path.join(cur_root, 'audio')) if item.endswith('.npy')]
    if len(cur_depth_ts) != 0 and len(cur_radar_ts) != 0 and len(cur_audio_ts) != 0:
        audio_ts, depth_ts, radar_ts = set(cur_audio_ts), set(cur_depth_ts), set(cur_radar_ts)
        common_ts = audio_ts.intersection(depth_ts, radar_ts)
        if yolo is not None:
            common_ts = common_ts.intersection(set(yolo))
        common_ts = list(common_ts)
    elif len(cur_depth_ts) != 0 and len(cur_radar_ts) != 0:
        depth_ts, radar_ts = set(cur_depth_ts), set(cur_radar_ts)
        common_ts = depth_ts.intersection(radar_ts)
        common_ts = list(common_ts)
    elif len(cur_depth_ts) != 0 and len(cur_audio_ts) != 0:
        depth_ts, audio_ts = set(cur_depth_ts), set(cur_audio_ts)
        common_ts = depth_ts.intersection(audio_ts)
        common_ts = list(common_ts)
    elif len(cur_depth_ts) != 0:
        common_ts = set(cur_depth_ts)
        common_ts = list(common_ts)
    else:
        print(f"Failed to find any data under {cur_root}")
        return

    common_ts = sorted(common_ts, key=lambda x: datetime.strptime(x, '%Y-%m-%d_%H-%M-%S'))
    df = pd.DataFrame({'id': np.arange(len(common_ts)), 'timestamp': common_ts})
    df.to_csv(os.path.join(target_root, "timestamp.csv"), index=False)
    
    with Pool(workers) as pool:
        for id, timestamp in enumerate(common_ts):
            pool.apply_async(_process_single_time, args=(cur_root, target_root, timestamp, id))
        pool.close()
        pool.join()
    

def run():
    sample_idx = [item for item in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, item)) and item != 'filtered_videos']
    process = []
    for id in sample_idx:
        yolo_ts = None
        cur_label_dir = os.path.join(data_path, id, 'label')
        target_label_dir = os.path.join(target_path, id, 'label')
        cur_unlabel_dir = os.path.join(data_path, id, 'unlabel')
        target_unlabel_dir = os.path.join(target_path, id, 'unlabel')
        process.append(Process(target=_process_single_subject, args=(cur_label_dir, target_label_dir, yolo_ts)))
        # process.append(Process(target=_process_single_subject, args=(cur_unlabel_dir, target_unlabel_dir, yolo_ts)))
    for p in process:
        p.start()
    for p in process:
        p.join()


if __name__ == '__main__':
    run()
