# AD-Data-Process

1. 先跑data_pre, 再跑data_split, data_pre有nx后缀是nx版本的数据，没有是rpi版本的数据，据说有一些不一样
2. 
Data sampling: first 20s as one sample every 200s​

 Duration 7:00-19:00 in 4 weeks​

 Label data: first 2s as one labeled sample every 20s (1% labelled data)​

 Unlabel data: last 18s as one unlabeled sample every 20s​

Yolo detection: depth images and delete the data samples without humans​

 Version: Yolo v5​

 Parameters: default: threshold 16 frame—3 frame​

Data dimension ​

 Depth: [16,112,112]​

 Radar (voxels) : [20,2,16,32,16]​

 Audio (Mel-frequency cepstral coefficients) : [20,87]
 
3. “ad的数据处理，是跑4周的数据，取的npz吗？”，
   “之前是一周的数据吧好像”，“我怎么记得只跑一天的7小时”
   “一周的，每天7小时，但你再看看代码，我也不敢乱说”
   “我们的标准是 每200秒取前20秒：20秒中前2s作为label data; 后18s作为unlabel data不”
   “其实是每20s取前两秒作为一个sample，每十个sample中按1:9分label与unlabel”
   “嗯，差不多”
