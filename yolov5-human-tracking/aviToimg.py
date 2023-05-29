import os
import numpy as np
import cv2
import sys

video_src_src_path = str(sys.argv[1])
print(video_src_src_path)
#video_src_src_path = 'NX1/labeldata/video'  # dataset path
label_name = os.listdir(video_src_src_path)
label_dir = {}
index = 0

for i in label_name:
    if i.startswith('.'):
        continue
    label_dir[i] = index
    index += 1
    video_src_path = os.path.join(video_src_src_path, i)
    #print(i)
    #print(video_src_path)
    video_save_path = video_src_src_path + '/img/' +i[:-4]
    #print(video_save_path)
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    videoFile =  video_src_path
    print(videoFile)
    outputFile = video_save_path
    #print(outputFile)
    vc = cv2.VideoCapture(videoFile)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        print('openerror!')
        rval = False
    timeF = 1  #视频帧计数间隔次数
    while rval:
        rval, frame = vc.read()
        if (c % timeF == 0 and c!=30):
            cv2.imwrite(outputFile+"/"+ str(int(c / timeF)) + '.jpg', frame)
        if (c==30):
            break
        c += 1
        cv2.waitKey(1)
    vc.release()
