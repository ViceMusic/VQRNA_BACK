# 一些工具类型

import json
import torch
import time
import random
import hashlib
from multiprocessing import Lock

# 将json字符串转化为序列对象的方法
def json_to_dict(json_str):
    return json.loads(json_str)

# 转化为张量, 不过这个好像用不上
def fasta_to_list(fasta_file   ):
    sequences_2d = []
    current_sequence = []

    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('>'):  # 跳过空行和header
                if current_sequence:  # 遇到新header时，保存当前序列
                    sequences_2d.append(current_sequence)
                    current_sequence = []
                continue

            # 将当前行的每个字符拆分为碱基列表
            current_sequence.extend(list(line.upper()))  # 统一转为大写

        # 添加最后一个序列
        if current_sequence:
            sequences_2d.append(current_sequence)

    return sequences_2d

# 将二维碱基list转化为张量(映射已经实现了)
def list_to_tensor(site_list):
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    numerical_list = [[mapping[base] for base in seq] for seq in site_list]
    tensor = torch.tensor(numerical_list)
    return tensor

# 生成随机名称的方法
def generate_unique_filename(extension: str = "", prefix: str = "", length: int = 12) -> str:
    """
        extension (str): 文件扩展名（如 ".txt"），默认无扩展名
        prefix (str): 文件名前缀（如 "data_"），默认为空
        length (int): 最终随机部分的显示长度（截取哈希值前N位）
    返回:
        str: 格式如 "prefix_a3c8b2d4.extension"
    """
    timestamp = int(time.time() * 1e6)  # 微秒级时间戳
    rand_num = random.randint(0, 0xFFFFFFFF)
    unique_seed = f"{timestamp}{rand_num}{random.getrandbits(128)}".encode()
    hash_digest = hashlib.sha1(unique_seed).hexdigest()
    unique_id = hash_digest[:length]

    filename = f"{prefix}{unique_id}{extension}"
    return filename

# fasta转化为seq_list
def fasta_to_seq_list(file_path: str, merge_lines: bool = False):
    sequences = []
    current_sequence = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if merge_lines and current_sequence:
                    sequences.append(''.join(current_sequence))
                    current_sequence = []
                continue

            if merge_lines:
                current_sequence.append(line)
            else:
                sequences.append(line)

    # 处理最后一个序列
    if merge_lines and current_sequence:
        sequences.append(''.join(current_sequence))

    return sequences

# 生成互斥锁, 创建跨进程的锁对象
lock = Lock()  # 基础互斥锁