import os
import json
import matplotlib.pyplot as plt


def plot_merged_data(input_dir, output_dir):
    """
    绘制 frame_data JSON 数据的折线图
    :param input_dir: 输入目录，包含 frame_data.json 文件
    :param output_dir: 输出目录，用于保存图像
    """
    # 遍历输入目录中的所有 JSON 文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith("frame_data.json"):  # 仅处理以 frame_data.json 结尾的文件
            input_path = os.path.join(input_dir, file_name)

            # 加载 JSON 数据
            with open(input_path, 'r') as file:
                data = json.load(file)

            if not data:
                print(f"File {file_name} is empty. Skipping...")
                continue

            # 提取每一维数据
            frames = range(len(data))  # 帧数
            pitch = [entry[0] for entry in data]
            yaw = [entry[1] for entry in data]
            roll = [entry[2] for entry in data]
            left_pitch = [entry[3] for entry in data]
            left_yaw = [entry[4] for entry in data]
            right_pitch = [entry[5] for entry in data]
            right_yaw = [entry[6] for entry in data]

            # 创建折线图
            plt.figure(figsize=(12, 6))
            plt.plot(frames, pitch, label="Pitch (Pose)", linestyle='-')
            plt.plot(frames, yaw, label="Yaw (Pose)", linestyle='--')
            plt.plot(frames, roll, label="Roll (Pose)", linestyle='-.')
            plt.plot(frames, left_pitch, label="Left Eye Pitch", linestyle=':')
            plt.plot(frames, left_yaw, label="Left Eye Yaw", linestyle='-')
            plt.plot(frames, right_pitch, label="Right Eye Pitch", linestyle='--')
            plt.plot(frames, right_yaw, label="Right Eye Yaw", linestyle='-.')

            # 图例和标签
            plt.title(f"7-Dimensional Merged Data for {file_name}")
            plt.xlabel("Frame Number")
            plt.ylabel("Values")
            plt.legend()
            plt.grid(True)

            # 保存图像
            id_name = file_name.replace("_frame_data.json", "")  # 提取 ID 名称
            output_path = os.path.join(output_dir, f"{id_name}_merged_plot.png")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path)
            plt.close()

            print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    # 配置路径
    input_dir = "assets/tracker_mpg_res"  # 包含 frame_data.json 的目录
    output_dir = "plots/tracker"  # 用于保存图像的目录

    # 绘制并保存折线图
    plot_merged_data(input_dir, output_dir)
