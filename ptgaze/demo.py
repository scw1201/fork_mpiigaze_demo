'''输出pose，gaze七维数据并存储'''
import datetime
import logging
import os
import pathlib
from typing import Optional
import cv2
import numpy as np
from omegaconf import DictConfig
from .common import Face, FacePartsName, Visualizer
from .gaze_estimator import GazeEstimator
from .utils import get_3d_face_model
import json
import subprocess
from scipy.spatial.transform import Rotation as R  # 确保导入 Rotation 模块
import imageio
import cv2
import pathlib
import moviepy.editor as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        # self.writer = self._create_video_writer()
        self.frame_data = []
        self.com_frame_img = []
        self.pred_frame_img = []

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def _save_frame_data(self, video_name: str) -> None:
        output_path = pathlib.Path(self.config.demo.output_dir) / f'{video_name}_frame_data.json'
        with open(output_path, 'w') as f:
            json.dump(self.frame_data, f)
        # print('saved.')

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)

        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    import os

    def _save_frames_as_video(self, frames, output_path, fps):
        # 创建一个临时视频文件以保存帧
        temp_video_path = pathlib.Path(output_path).with_name(pathlib.Path(output_path).stem + '_temp.mp4')

        io_writer = imageio.get_writer(str(temp_video_path), fps=fps)  # 转换为字符串

        for frame in frames:
            # 将 BGR 帧转换为 RGB 帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            io_writer.append_data(frame_rgb)

        io_writer.close()

        # 使用 moviepy 将原视频的音频合并到新视频中
        original_video = mp.VideoFileClip(str(self.config.demo.video_path))  # 转换为字符串
        temp_video = mp.VideoFileClip(str(temp_video_path))  # 转换为字符串

        # 设置输出视频的音频
        final_video = temp_video.set_audio(original_video.audio)

        # 写入最终的视频文件
        final_video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')  # 转换为字符串

        # 关闭视频文件
        temp_video.close()
        original_video.close()

        # 删除临时视频文件
        os.remove(str(temp_video_path))

    def _run_on_video(self) -> None:
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            original_frame = frame.copy()  # 保留原始帧的副本
            self._process_image(frame)  # 处理帧

            # 将原始帧和处理后的帧左右拼接
            combined_frame = np.hstack((original_frame, self.visualizer.image))

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', combined_frame)

            # 保存拼接后的帧
            self.com_frame_img.append(combined_frame)
            self.pred_frame_img.append(self.visualizer.image)

        # 使用 imageio 将帧列表写入视频文件
        video_name = pathlib.Path(self.config.demo.video_path).stem
        line_output_path = pathlib.Path(self.config.demo.output_dir) / f"{video_name}_line.mp4"
        both_output_path = pathlib.Path(self.config.demo.output_dir) / f"{video_name}_side_by_side.mp4"
        self._save_frames_as_video(self.com_frame_img, both_output_path, fps=25)
        self._save_frames_as_video(self.pred_frame_img, line_output_path, fps=25)

        self.cap.release()
        cv2.destroyAllWindows()

        # 保存每个视频的帧数据
        video_name = pathlib.Path(self.config.demo.video_path).stem
        self._save_frame_data(video_name)


    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(image, self.gaze_estimator.camera.camera_matrix, self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)

        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)

            # 调用 _draw_head_pose 并获取当前帧的姿势列表
            now_pose_list,rotation_head = self._draw_head_pose(face)
            # 调用 _draw_gaze_vector 并获取当前帧的注视方向列表
            gaze_directions = self._draw_gaze_vector(face, rotation_head)
            now_pose_list = list(now_pose_list)
            gaze_directions = list(gaze_directions)

            # 将两个列表合并为一个 7 维向量
            combined_list = now_pose_list + gaze_directions

            # print('combined_list:',combined_list)
            # print('combined_length',len(combined_list))
            if combined_list:
                self.frame_data.append(combined_list)

            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        # if self.writer:
        #     self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')


    def create_video_using_ffmpeg(input_frames, output_path, codec='libx264', fps=25):
        # output_path 是视频文件的输出路径
        # codec 是视频编码器，例如 'libx264' 或 'mpeg4'
        # fps 是帧率

        # 使用 FFmpeg 将帧写入视频
        cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',  # 确保使用正确的像素格式
            '-s', f'{256}x{256}',  # 替换 width 和 height 为视频的分辨率
            '-r', str(fps),
            '-i', '-',
            '-an',  # 不需要音频
            '-c:v', codec,
            '-pix_fmt', 'yuv420p',  # 确保使用正确的像素格式
            output_path
        ]
        with open(input_frames, 'rb') as pipe:
            subprocess.run(cmd, stdin=pipe)

    # 示例使用
    # create_video_using_ffmpeg('input_frames.txt', 'output.mp4')
    # def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
    #     print('writing video......')
    #     if self.config.demo.image_path:
    #         return None
    #     if not self.output_dir:
    #         return None
    #     ext = self.config.demo.output_file_extension
    #     if ext == 'mp4':
    #         fourcc = cv2.VideoWriter_fourcc(*'H264')
    #     elif ext == 'avi':
    #         fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    #     else:
    #         raise ValueError
    #     if self.config.demo.use_camera:
    #         output_name = f'{self._create_timestamp()}.{ext}'
    #     elif self.config.demo.video_path:
    #         name = pathlib.Path(self.config.demo.video_path).stem
    #         output_name = f'{name}.{ext}'
    #     else:
    #         raise ValueError
    #     output_path = self.output_dir / output_name
    #     writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 25,
    #                              (self.gaze_estimator.camera.width,
    #                               self.gaze_estimator.camera.height))
    #     if writer is None:
    #         raise RuntimeError
    #     return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> list:

        if not self.show_head_pose:
            return []
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True) # head_pose_rot是相机坐标系（相机正对头部）下的头动旋转矩阵
        pitch, yaw, roll = face.change_coordinate_system(euler_angles) # 世界坐标系的头动角度

        # 将欧拉角转换为旋转矩阵
        rotation_world = R.from_euler('XYZ', [pitch, yaw, roll], degrees=True)
        head_rotation_matrix_world = rotation_world.as_matrix()  # 3x3 旋转矩阵

        # logger.info(f'pose_list_{pitch, yaw, roll}')
        model_coor_euler_angles = [pitch, yaw, roll]
        return model_coor_euler_angles, head_rotation_matrix_world

    def _draw_gaze_vector(self, face: Face, rotation_head) -> list:
        gaze_directions = []
        length = self.config.demo.gaze_visualization_length

        if self.config.mode == 'MPIIGaze':
            # 处理左右眼
            for key in [FacePartsName.LEYE, FacePartsName.REYE]:
                eye = getattr(face, key.name.lower())

                # 画注视向量
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)

                # 使用传入的旋转矩阵进行计算
                gaze_vector_head = np.linalg.inv(face.head_pose_rot.as_matrix()) @ eye.gaze_vector
                # 计算模型坐标系下的 pitch 和 yaw
                # print('gaze_vec',eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(gaze_vector_head))

                # 根据眼睛的顺序，先是左眼，后是右眼
                if key == FacePartsName.LEYE:
                    gaze_directions.extend([pitch, yaw])  # 左眼的世界坐标系 pitch 和 yaw
                else:
                    gaze_directions.extend([pitch, yaw])  # 右眼的世界坐标系 pitch 和 yaw

            # logger.info(f'gaze_list: {gaze_directions}')
            return gaze_directions


        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            # 将 face.gaze_vector 从相机坐标系转换为模型坐标系
            rot_matrix = face.head_pose_rot.as_matrix()
            gaze_vector_in_model = np.linalg.inv(rot_matrix).dot(face.gaze_vector)

            # 画注视向量
            self.visualizer.draw_3d_line(
                face.center, face.center + length * gaze_vector_in_model)

            # 计算模型坐标系下的 pitch 和 yaw
            pitch, yaw = np.rad2deg(face.vector_to_angle(gaze_vector_in_model))

            # 如果模式为 'MPIIFaceGaze' 或 'ETH-XGaze'，只返回一个点的 pitch 和 yaw
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
            return [pitch, yaw, None, None]  # 返回一个包含四个值的列表，右眼值为 None

        else:
            raise ValueError

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)
