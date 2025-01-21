
-捕捉到的raw数据放进来，跑process_tracker_video.py，进行帧率对齐到25fps和分辨率统一，处理好的视频存储在assets/tracker_gt_video
-跑python batch_ptgaze.py --folder '/media/lenovo/本地磁盘/1_talkingface/Audio2GP/train_data/TFHP_cropped' --output-dir '/media/lenovo/本地磁盘/1_talkingface/Audio2GP/train_data/TFHP_mpg_res' --mode mpiigaze，识别结果在assets/tracker_mpg_res
-norm到第一帧：norm.py
-画图plot.py