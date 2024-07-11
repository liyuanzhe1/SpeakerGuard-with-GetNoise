import os
import shutil

def create_file_dict(directory):
    file_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):  # 根据实际文件类型进行过滤
            name_part = filename.rsplit('.', 1)[0]  # 去掉文件后缀
            parts = name_part.split('-')[-1].split('_')
            abcd = parts[0]
            x = parts[1]
            x_padded = x.zfill(4)
            file_dict[x_padded] = abcd
            
    return file_dict

def rename_files_with_dict(source_directory, target_directory, file_dict):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for filename in os.listdir(source_directory):
        if filename.endswith('.wav'):  # 根据实际文件类型进行过滤
            name_part = filename.rsplit('.', 1)[0]  # 去掉文件后缀
            parts = name_part.split('-')[-1].split('_')
            abcd = parts[0]
            if abcd in file_dict.values():
                # 找到匹配的 key
                matching_key = [key for key, value in file_dict.items() if value == abcd][0]
                new_filename = filename.replace(abcd, matching_key)
                src_path = os.path.join(source_directory, filename)
                dest_path = os.path.join(target_directory, new_filename)
                shutil.copy(src_path, dest_path)  # 将文件复制到新目录并重命名

# 使用方法
original_directory = '/home/adgroup2/mnt1/xvectors/SpeakerGuard/adv_output/fakebob/302'  # 替换为存储302-123504-abcd_x.wav文件的文件夹路径
source_directory = '/home/adgroup2/mnt1/xvectors/SpeakerGuard/adv_output/noise/fakebob'  # 替换为存储302-123504-abcd_noise.wav文件的文件夹路径
target_directory = '/mnt/hyh/adv_output/noise/fakebob'  # 替换为你希望保存重命名文件的文件夹路径

file_dict = create_file_dict(original_directory)
rename_files_with_dict(source_directory, target_directory, file_dict)
