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
    source_roots = [
        'data/honor_coll_data/honor_data_0920/20250711_1519_finterval1',
        'data/honor_coll_data/honor_data_0920/20250711_1540_finterval1'
    ]  
    # 保存文件路径到txt文件
    output_dir = './data/honor_coll_data/honor_data_0920/'
    os.makedirs(output_dir, exist_ok=True)
    
    preprocess_folders = ['voxels']
    train_files = []
    test_files = []
    
    # 3. 处理每个源文件夹
    for source_root in source_roots:
        # 获取源文件夹名称作为前缀
        source_name = os.path.basename(source_root)
        
        # 计算当前源文件夹的分割值（使用voxels文件夹中的pkl文件）
        voxels_path = os.path.join(source_root, preprocess_folders[0])
        split_value = split_train_and_test(voxels_path)
        
        # 遍历当前源文件夹下的子文件夹
        for folder_name in os.listdir(source_root):
            if folder_name not in preprocess_folders:
                continue
            folder_path = os.path.join(source_root, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
                
            # 处理当前文件夹中的文件
            for filename in os.listdir(folder_path):
                if not filename.endswith(".pkl"):
                    continue
                    
                try:
                    # 提取文件名中的数字部分
                    name_part = os.path.splitext(filename)[0]
                    file_num = int(name_part)
                    
                    # 构建相对路径（如：20250711_1519_finterval1/voxels/000600.pkl）
                    relative_path = f"{source_name}/{folder_name}/{filename}"
                    
                    # 根据分割值划分训练集和测试集
                    if file_num <= split_value:
                        train_files.append(relative_path)
                    else:
                        test_files.append(relative_path)
                        
                except ValueError:
                    continue

    # 保存训练集文件路径
    with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
        for path in train_files:
            f.write(f"{path}\n")
    
    # 保存测试集文件路径
    with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
        for path in test_files:
            f.write(f"{path}\n")

if __name__ == "__main__":
    # 执行处理
    process_honor_files()
    print("honor_coll_data文件处理完成！")