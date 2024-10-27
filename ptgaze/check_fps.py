import cv2
import pathlib

folder_path = pathlib.Path('/media/lenovo/本地磁盘/1_talkingface/mapiigaze/fork_mpiigaze_demo/assets/mpg_pipeline_input')
video_files = list(folder_path.glob('*.mp4')) + list(folder_path.glob('*.avi'))
for file in video_files:
    cap = cv2.VideoCapture(file)

    if cap.isOpened():
        # 获取帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{file}: {fps} FPS")
    else:
        print("无法打开视频文件")

    # 释放资源
    cap.release()
