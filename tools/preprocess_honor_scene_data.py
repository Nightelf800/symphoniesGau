import os
import shutil


def split_train_and_test(folder_path):
    file_numbers = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理后缀为.pkl的文件
        if filename.endswith(".pkl"):
            # 提取文件名前缀（去掉.pkl），并尝试转为整数
            try:
                # 假设文件名格式是“6位数字.pkl”，如000000.pkl → 提取“000000”
                num_str = filename.split(".")[0]  # 分割文件名，取“.”前面的部分
                num = int(num_str)  # 转为整数（如“000001”→1，不影响排序）
                file_numbers.append(num)
            except:
                # 跳过非“数字.pkl”格式的文件（如abc.pkl）
                print(f"跳过非目标文件：{filename}")

    # -------------------------- 2. 排序并计算分界值 --------------------------
    if not file_numbers:
        print("文件夹中未找到符合格式的.pkl文件！")
    else:
        # 1. 从小到大排序
        sorted_numbers = sorted(file_numbers)
        # 2. 去重（若有重复文件，保留一个）
        sorted_unique = list(set(sorted_numbers))
        sorted_unique.sort()  # 去重后重新排序（确保顺序正确）
        N = len(sorted_unique)  # 有效数据总数

        # 3. 计算80%分界位置（取整数，向下取整）
        split_index = int(N * 0.8)
        # 注意：若split_index=0（如N=1），则所有数据属于前80%
        if split_index == 0:
            split_value = 0
            print(f"数据总数过少（仅{N}个），无法划分前80%与后20%")
        else:
            # 分界值 = 前80%的最后一个数（或后20%的第一个数-1，根据需求表述）
            split_value = sorted_unique[split_index - 1]  # 前80%的最后一个数
            next_value = sorted_unique[split_index] if split_index < N else "无"  # 后20%的第一个数
        return split_value


def process_honor_files():
    # 源目录和目标目录设置
    source_root = './data/honor_data_1579'
    preprocess_folders = [
        'color',
        'depth',
        'voxels'
    ]
    target_train_root = './data/honor_data_1579/honor_sample1579_0904/train'
    target_test_root = './data/honor_data_1579/honor_sample1579_0904/test'

    # 确保目标目录存在
    os.makedirs(target_train_root, exist_ok=True)
    os.makedirs(target_test_root, exist_ok=True)

    split_value = split_train_and_test(os.path.join(source_root, preprocess_folders[2]))

    # 遍历源目录下的所有文件夹
    for folder_name in os.listdir(source_root):
        if folder_name not in preprocess_folders:
            continue
        folder_path = os.path.join(source_root, folder_name)

        # 只处理目录
        if not os.path.isdir(folder_path):
            continue

        # 创建对应的train和test子目录
        train_folder = os.path.join(target_train_root, folder_name)
        test_folder = os.path.join(target_test_root, folder_name)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # 处理当前文件夹中的文件
        for filename in os.listdir(folder_path):
            # 尝试从文件名提取数字部分（假设文件名格式为000000.jpg等）
            try:
                # 分离文件名和扩展名
                name_part, ext_part = os.path.splitext(filename)
                # 转换为数字
                file_num = int(name_part)

                # 判断文件应该复制到哪个目录
                src_path = os.path.join(folder_path, filename)
                if 0 <= file_num <= split_value:
                    dest_path = os.path.join(train_folder, filename)
                    shutil.copy2(src_path, dest_path)
                elif file_num > split_value:
                    dest_path = os.path.join(test_folder, filename)
                    shutil.copy2(src_path, dest_path)

            except ValueError:
                # 如果无法提取数字，则跳过该文件
                continue


if __name__ == "__main__":
    # 执行处理
    process_honor_files()
    print("honor_coll_data文件处理完成！")