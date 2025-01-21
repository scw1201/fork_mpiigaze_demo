import os
import json


def process_json_file(file_path, output_dir):
    """
    处理单个 JSON 文件，以第一帧的前三维为基准计算每帧的前三维差值。
    :param file_path: 输入 JSON 文件路径
    :param output_dir: 输出目录路径
    """
    # 读取 JSON 文件
    with open(file_path, 'r') as f:
        data = json.load(f)

    if not data:
        print(f"File {file_path} is empty or invalid.")
        return

    # 以第一帧为基准
    first_frame = data[0]
    adjusted_data = []

    for frame in data:
        adjusted_frame = [
            current - base if idx < 3 else current
            for idx, (current, base) in enumerate(zip(frame, first_frame))
        ]
        adjusted_data.append(adjusted_frame)

    # 保存为新的 JSON 文件
    file_name = os.path.basename(file_path).replace("_final_processed_frame_data.json", "_normed.json")
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w') as f:
        json.dump(adjusted_data, f, indent=4)

    print(f"Processed file saved to: {output_path}")


def process_all_json_files(input_dir, output_dir):
    """
    批量处理一个文件夹下的所有 JSON 文件。
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            process_json_file(file_path, output_dir)


if __name__ == "__main__":
    # 输入和输出目录
    input_dir = "assets/tracker_mpg_res"  # 包含所有 JSON 文件的目录
    output_dir = "assets/tracker_mpg_res_normed"  # 输出处理后 JSON 文件的目录

    process_all_json_files(input_dir, output_dir)
