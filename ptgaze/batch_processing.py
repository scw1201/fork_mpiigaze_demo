import os
import subprocess
import argparse
import pathlib


def process_folder(video_folder, config_path, mode, face_detector, device, camera, output_dir, ext, no_screen, debug):
    video_folder = pathlib.Path(video_folder)
    for video_file in video_folder.glob('*.mp4'):
        process_video(video_file, config_path, mode, face_detector, device, camera, output_dir, ext, no_screen, debug)


def process_video(video_path, config_path, mode, face_detector, device, camera, output_dir, ext, no_screen, debug):
    command = [
        'python', 'ptgaze/main.py',  # 修改为实际路径
        '--config', config_path,
        '--video', video_path.as_posix(),
        '--mode', mode
    ]

    if face_detector:
        command.extend(['--face-detector', face_detector])
    if device:
        command.extend(['--device', device])
    if camera:
        command.extend(['--camera', camera])
    if output_dir:
        command.extend(['--output-dir', output_dir])
    if ext:
        command.extend(['--ext', ext])
    if no_screen:
        command.append('--no-screen')
    if debug:
        command.append('--debug')

    print(f"Running command: {' '.join(command)}")  # 打印命令以进行调试

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error processing {video_path}: {result.stderr}")
    else:
        print(f"Successfully processed {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos using main.py")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--video_folder', type=str, required=True, help='Path to the folder containing videos')
    parser.add_argument('--mode', type=str, required=True, help='Mode for processing videos')
    parser.add_argument('--face-detector', type=str, help='Face detector to use')
    parser.add_argument('--device', type=str, help='Device to use for inference')
    parser.add_argument('--camera', type=str, help='Path to camera calibration file')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory for the results')
    parser.add_argument('--ext', '-e', type=str, choices=['avi', 'mp4'], help='Output video file extension')
    parser.add_argument('--no-screen', action='store_true', help='Do not display video on screen')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    process_folder(args.video_folder, args.config, args.mode, args.face_detector, args.device, args.camera,
                   args.output_dir, args.ext, args.no_screen, args.debug)
