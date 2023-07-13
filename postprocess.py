import os
import cv2
import numpy as np
from nx_samplers import Pool
from multiprocessing import Process


data_path = "/pm1733_x3/sample_li"
filter_path = "/pm1733_x3/filtered_videos"
target_path = "/pm1733_x3/features_li"
depth_frames = 16
depth_shape = (16, 112, 112)
radar_shape = (20, 2, 16, 32, 16)
audio_shape = (20, 87)
max_workers = 32

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

def _make_modality_path(root):
    modal = ['audio', 'dehtp', 'radar']
    for m in modal:
        if not os.path.exists(os.path.join(root, m)):
            os.makedirs(os.path.join(root, m))

def _process_single_time(source_root, target_root, timestamp):
    try:
        _make_modality_path(target_root)
        audio_path = os.path.join(source_root, 'audio')
        depth_path = os.path.join(source_root, 'depth')
        radar_path = os.path.join(source_root, 'radar')
        audio_file = np.load(os.path.join(audio_path, timestamp+'.npy'))
        radar_file = np.load(os.path.join(radar_path, timestamp+'.npy'))
        depth_features = process_video(os.path.join(depth_path, timestamp+'.mp4'))
        if audio_file.shape == audio_shape and radar_file.shape == radar_shape and depth_features == depth_shape:
            np.save(os.path.join(target_root, 'audio', timestamp+'.npy'), audio_file)
            np.save(os.path.join(target_root, 'depth', timestamp+'.npy'), depth_features)
            np.save(os.path.join(target_root, 'radar', timestamp+'.npy'), radar_file)
    except Exception as e:
        print(f"Failed to process {timestamp} data under {source_root}. Error message {e}")


def _process_single_subject(cur_root, target_root, yolo=None):
    cur_depth_ts = [item.split('.')[0] for item in os.listdir(os.path.join(cur_root, 'depth')) if item.endswith('.mp4')]
    cur_radar_ts = [item.split('.')[0] for item in os.listdir(os.path.join(cur_root, 'radar')) if item.endswith('.npy')]
    cur_audio_ts = [item.split('.')[0] for item in os.listdir(os.path.join(cur_root, 'audio')) if item.endswith('.npy')]
    audio_ts, depth_ts, radar_ts = set(cur_audio_ts), set(cur_depth_ts), set(cur_radar_ts)
    common_ts = audio_ts.intersection(depth_ts, radar_ts)
    if yolo is not None:
        common_ts = common_ts.intersection(set(yolo))
    process_list = []
    for timestamp in common_ts:
        process_list.append(Process(target=_process_single_time, args=(cur_root, target_root, timestamp)))
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()
    

def run(num_workers):
    filter_idx = [item for item in os.listdir(filter_path) if os.path.isdir(os.path.join(filter_path, item))]
    sample_idx = [item for item in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, item))]
    with Pool(num_workers) as pool:
        for id in sample_idx:
            if id in filter_idx:
                yolo_ts = [item.split('.')[0] for item in os.listdir(os.path.join(filter_path, id)) if item.endswith('.mp4')]
            else:
                yolo_ts = None
            cur_label_dir = os.path.join(data_path, id, 'label')
            target_label_dir = os.path.join(target_path, id, 'label')
            cur_unlabel_dir = os.path.join(data_path, id, 'unlabel')
            target_unlabel_dir = os.path.join(target_path, id, 'unlabel')
            pool.apply(_process_single_subject, args=(cur_label_dir, target_label_dir, yolo_ts))
            pool.apply(_process_single_subject, args=(cur_unlabel_dir, target_unlabel_dir))
        pool.close()
        pool.join()


if __name__ == '__main__':
    run(max_workers)
