import os
import subprocess

"""
遍历输入目录下的所有子文件夹，处理 AVI 视频文件，将分辨率调整为 640:640，帧率改为 25fps。

:param input_dir: 输入目录，包含原始 AVI 视频。
:param output_dir: 输出目录，保存处理后的视频。
:param resolution: 目标分辨率，默认为 640:640。
:param fps: 目标帧率，默认为 25 fps。
"""

def process_videos(input_dir, output_dir, resolution="640:640", fps=25):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith("_final.avi"):  # 仅处理文件名包含 `_final.avi` 的视频
                input_path = os.path.join(root, file_name)

                # 输出文件路径：将所有处理结果存储在 output_dir
                output_file_name = file_name.replace("_final.avi", ".mp4")
                output_path = os.path.join(output_dir, output_file_name)

                # FFmpeg 命令
                command = [
                    "ffmpeg",
                    "-err_detect", "ignore_err",  # 忽略错误
                    "-i", input_path,
                    "-vf", f"scale={resolution}",
                    "-r", str(fps),
                    "-c:v", "libx264",  # 输出为 MP4 格式
                    "-crf", "23",  # 控制质量（数值越低质量越高）
                    "-preset", "fast",  # 编码速度
                    "-c:a", "aac",  # 使用 AAC 音频编码
                    "-b:a", "128k",  # 音频码率
                    output_path
                ]

                # 执行 FFmpeg 命令
                try:
                    subprocess.run(command, check=True)
                    print(f"Processed video saved to: {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    # 输入和输出目录
    input_directory = "raw_1_2"
    output_directory = "assets/tracker_gt_video"

    process_videos(input_directory, output_directory)
