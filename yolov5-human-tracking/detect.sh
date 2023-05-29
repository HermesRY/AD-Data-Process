#!/bin/bash
#NX1/labeldata/video
python3 aviToimg.py $1
for dir in $(ls $1/img)
do
  echo $1/img/$dir
  python3 detect.py --source $1/img/$dir >> detect.log
done

i=0
for dire in $(ls runs/detect)
do
  #echo $i
  #echo runs/detect/$dire
  size=$(du -sh runs/detect/$dire|cut -b 1);
  #echo $size;
  if (($size>0))
  then
    #rm $1/video_
    echo $1/video_$i.avi
  else
    rm $1/video_$i.avi
  fi
  i=$[i+10]
done
rm -r $1/img
rm -r runs/detect/*
