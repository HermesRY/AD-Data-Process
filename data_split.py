import numpy as np
import cv2
import os
import argparse
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, nargs='+')
    parser.add_argument("--raw_data_root", type=str, default="/mnt/nas")
    parser.add_argument("--save_data_root", type=str, default="/mnt/AD-temp-data")
    args = parser.parse_args()
    subjects = args.subjects
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    for subject in subjects:
        print(subject)
        radar_path = os.path.join(args.raw_data_root, str(subject), "radar")
        audio_path = os.path.join(args.raw_data_root, str(subject), "audio")
        depth_path = os.path.join(args.raw_data_root, str(subject), "depth")
        save_path = os.path.join(args.save_data_root, str(subject))
        if not os.path.exists(save_path):
            #os.makedirs(os.path.join(save_path, "label_data", "audio"))
            #os.makedirs(os.path.join(save_path, "label_data", "radar"))
            #os.makedirs(os.path.join(save_path, "label_data", "depth"))
            os.makedirs(os.path.join(save_path, "label_data", "feature"))
            os.makedirs(os.path.join(save_path, "label_data", "video"))
            os.makedirs(os.path.join(save_path, "unlabel_data", "feature"))
            #os.makedirs(os.path.join(save_path, "unlabel_data", "audio"))
            #os.makedirs(os.path.join(save_path, "unlabel_data", "radar"))
            #os.makedirs(os.path.join(save_path, "unlabel_data", "depth"))

        total_npy_num = len(os.listdir(depth_path))
        
        radar_label_data = []
        depth_label_data = []
        audio_label_data = []
        index_label_data = []
        label_len = 0
        label_file_id = 0
        
        radar_unlabel_data = []
        depth_unlabel_data = []
        audio_unlabel_data = []
        index_unlabel_data = []
        unlabel_len = 0
        unlabel_file_id = 0
        
        for idx in tqdm(range(total_npy_num)):
	    # label data
            if idx%10==0:
                index_label_data.append(idx)
                label_len += 1
                depth_arr = np.load(os.path.join(depth_path, f"depth_{str(idx)}.npy"))
                radar_label_data.append(np.load(os.path.join(radar_path, f"radar_{str(idx)}.npy")))
                depth_label_data.append(depth_arr)
                audio_label_data.append(np.load(os.path.join(audio_path, f"audio_{str(idx)}.npy")))
                depth_arr = np.array(depth_arr, dtype=np.uint8)
                video = cv2.VideoWriter(os.path.join(save_path, "label_data", "video", f"video_{str(idx)}.avi"), fourcc, 15.0, (480, 640), False)
                for frame in depth_arr:
                    video.write(frame)
                video.release()
                if label_len >= 1000:
                    np.savez_compressed(os.path.join(save_path, "label_data", "feature", str(label_file_id)), index=np.array(index_label_data), radar=np.array(radar_label_data), audio=np.array(audio_label_data), depth=np.array(depth_label_data))
                    radar_label_data = []
                    depth_label_data = []
                    audio_label_data = []
                    index_label_data = []
                    label_len = 0
                    label_file_id += 1
            # unlabel data
            else:
                index_unlabel_data.append(idx)
                unlabel_len += 1
                radar_unlabel_data.append(np.load(os.path.join(radar_path, f"radar_{str(idx)}.npy")))
                depth_unlabel_data.append(np.load(os.path.join(depth_path, f"depth_{str(idx)}.npy")))
                audio_unlabel_data.append(np.load(os.path.join(audio_path, f"audio_{str(idx)}.npy")))
                if unlabel_len >= 1000:
                    np.savez_compressed(os.path.join(save_path, "unlabel_data", "feature", str(unlabel_file_id)), index=np.array(index_unlabel_data), radar=np.array(radar_unlabel_data), audio=np.array(audio_unlabel_data), depth=np.array(depth_unlabel_data))
                    radar_unlabel_data = []
                    depth_unlabel_data = []
                    audio_unlabel_data = []
                    index_unlabel_data = []
                    unlabel_len = 0
                    unlabel_file_id += 1
        if label_len > 0:
            np.savez_compressed(os.path.join(save_path, "label_data", "feature", str(label_file_id)), index=np.array(index_label_data), radar=np.array(radar_label_data), audio=np.array(audio_label_data), depth=np.array(depth_label_data))
            radar_label_data = []
            depth_label_data = []
            audio_label_data = []
            index_label_data = []
            label_len = 0
        if unlabel_len > 0:
            np.savez_compressed(os.path.join(save_path, "unlabel_data", "feature", str(unlabel_file_id)), index=np.array(index_unlabel_data), radar=np.array(radar_unlabel_data), audio=np.array(audio_unlabel_data), depth=np.array(depth_unlabel_data))
            radar_unlabel_data = []
            depth_unlabel_data = []
            audio_unlabel_data = []
            index_unlabel_data = []
            unlabel_len = 0
