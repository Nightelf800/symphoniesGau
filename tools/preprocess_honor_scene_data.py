import os
import shutil


def process_honor_files():
    # 源目录和目标目录设置
    source_root = './data/honor_collection_data'
    target_train_root = './data/honor_collection_data/train'
    target_test_root = './data/honor_collection_data/test'
    
    # 确保目标目录存在
    os.makedirs(target_train_root, exist_ok=True)
    os.makedirs(target_test_root, exist_ok=True)
    
    # 遍历源目录下的所有文件夹
    for folder_name in os.listdir(source_root):
        folder_path = os.path.join(source_root, folder_name)
        
        # 只处理目录
        if not os.path.isdir(folder_path):
            continue
        if folder_name == 'train' or folder_name == 'test':
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
                if 0 <= file_num <= 649:
                    dest_path = os.path.join(train_folder, filename)
                    shutil.copy2(src_path, dest_path)
                elif file_num >= 650:
                    dest_path = os.path.join(test_folder, filename)
                    shutil.copy2(src_path, dest_path)
                    
            except ValueError:
                # 如果无法提取数字，则跳过该文件
                continue

if __name__ == "__main__":
    # 执行处理
    process_honor_files()
    print("honor_coll_data文件处理完成！")