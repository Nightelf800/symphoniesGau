import os
import shutil

# 定义源目录和目标目录
base_src_dir = '/mnt/bn/yuanlichen0610modeleval/datasets/syndata/test'
base_dst_dir = '/mnt/bn/yuanlichen0610modeleval/datasets/syndata/data'

# 定义子文件夹列表
sub_folders = ['color', 'depth', 'intrinsic', 'output_occ', 'pose']

# 创建目标文件夹
for sub_folder in sub_folders:
    train_dir = os.path.join(base_dst_dir, 'train', sub_folder)
    test_dir = os.path.join(base_dst_dir, 'test', sub_folder)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

# 复制文件
for sub_folder in sub_folders:
    src_folder = os.path.join(base_src_dir, sub_folder)
    train_dst_folder = os.path.join(base_dst_dir, 'train', sub_folder)
    test_dst_folder = os.path.join(base_dst_dir, 'test', sub_folder)

    # 获取源文件夹中的所有文件并排序
    files = sorted(os.listdir(src_folder))
    for i, file in enumerate(files, start=1):
        src_file_path = os.path.join(src_folder, file)
        if i < 80:
            dst_file_path = os.path.join(train_dst_folder, file)
        else:
            dst_file_path = os.path.join(test_dst_folder, file)
        shutil.copy2(src_file_path, dst_file_path)

print("文件复制完成")