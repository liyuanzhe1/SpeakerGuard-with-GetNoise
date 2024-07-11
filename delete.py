import os
import shutil

def remove_suffix_and_copy(source_directory, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for filename in os.listdir(source_directory):
        if filename.endswith('.wav'):  # 仅处理.wav文件
            new_filename = filename.split('_')[0] + '.wav'  # 移除_x部分
            src_path = os.path.join(source_directory, filename)
            dest_path = os.path.join(target_directory, new_filename)
            shutil.copy(src_path, dest_path)  # 复制文件并重命名

# 使用方法
source_directory = '/home/adgroup2/mnt1/xvectors/SpeakerGuard/adv_output/fakebob/302'  # 替换为存储302-123504-abcd_x.wav文件的文件夹路径
target_directory = '/mnt/hyh/adv_output/adv_audio/fakebob'  # 替换为你希望保存重命名文件的文件夹路径

remove_suffix_and_copy(source_directory, target_directory)
