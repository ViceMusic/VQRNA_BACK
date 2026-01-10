# 这个文件，主要是用来配置模型内容的
# 因此在这里主要分为三个阶段
# 1.模型正常引入，加载，并且按照51的长度进行预测结果输出，应该是预测一个位点
# 2.引入十个模型，并且对同一个位点进行预测，再将这些收敛为一个方法，即为，调用十种模型，预测该位点的结果？

import torch
import torch.nn.functional as F
import sys
import os
import random
import itertools
import numpy as np
from flask import Flask, request, jsonify
import json
import base64
from PIL import Image
from io import BytesIO
from tangermeme.plot import plot_logo
from tangermeme.ism import saturation_mutagenesis
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("./"))  # 加载绘制motif的文件
from .Utils import *

print("导入绘图模块成功")


# 导入模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "models")

sys.path.append(MODEL_DIR)
from mymodel import Lucky
from mymodel_51 import Lucky as Lucky_51
# 设置设备类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载尺寸为51的模型，使用这个模型作为滑动窗口的内容
model_51 = Lucky_51().to(device)


# 导入模型参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAS_DIR = os.path.join(BASE_DIR, "..", "paras", "mymodel_51")

# 接下来需要完成的任务包括：
# 1. 对输入进行调试，确保能输入正常序列并且排错
# 2. 确定返回值类型，如果中途有失败情况怎么办
# 3. 绘制图片的函数需要重新进行一下处理，将其与单纯的预测进行合并，然后保证输出结果能直接转化base64格式用于后续内容
# 4. 返回成items的形式{ seq（序列）， type（数组，代表发生了何种甲基化），motif（图片），三元组 }三种信息，返回值先做成一个dict输出一下看看
# 5. 保持后端安全性，确保如果某个步骤不存在，那么就需要返回空或者{}/[]

# 关于之前的一个场景
# try-catch应该在外面还是在里面



# 对碱基进行映射操作

# 直接将一整个字符串转化为张量格式,"ACGC"-->[0,1,2,3]
def encode_dna_tensor(sequence):
    # 碱基映射规则,我全听师姐的，T=U, 但是我不保证数据是正常的，所以这里无论是T还是U都会被映射成为1
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3,'U':1}
    # 将 DNA 序列映射到索引
    encoded = [mapping[base] for base in sequence]
    # 转换为 PyTorch 张量，并调整形状为 (1, len(sequence))
    return torch.tensor(encoded).unsqueeze(0)

# 面向审稿人编程
# 面向审稿人编程(划掉), 用来检测是否发生了某些不对的情况
def pan(site, type_des):
    # 定义修饰与碱基的对应关系
    mod_to_base = {
        1: 'A',  # Am (2'-O-甲基腺苷)
        2: 'C',  # Cm (2'-O-甲基胞苷)
        3: 'G',  # Gm (2'-O-甲基鸟苷)
        4: 'U',  # Um (2'-O-甲基尿苷)
        5: 'A',  # m1A (1-甲基腺苷)
        6: 'C',  # m5C (5-甲基胞苷)
        7: 'U',  # m5U (5-甲基尿苷)
        8: 'A',  # m6A (6-甲基腺苷)
        9: 'A',  # m6Am (N6,2'-O-二甲基腺苷)
        10: 'U'  # Ψ (假尿苷)
    }

    # 检查 type_des 是否在有效范围内
    if type_des < 1 or type_des > 10:
        return False

    # 获取修饰对应的碱基
    expected_base = mod_to_base[type_des]

    # 检查传入的碱基是否与修饰对应
    if site.upper() != expected_base:
        return False

    # 如果都匹配，返回 True
    return True

# -------------绘图函数的准备区域---------------

# 进行碱基字符串裁剪，将字符串的长度向下取整即可，方便绘图统一
def trim_seq(seq, length):
    if(length<51):
        return "" # 如果返回空序列, 代表目前的长度不太行
    elif(length>=51 and length<101):
        return seq[:51]
    elif(length>=101 and length<201):
        return seq[:101]
    elif(length>=201 and length<301):
        return seq[:301]
    elif(length>=301 and length<401):
        return seq[:301]
    elif(length>=401 and length<501):
        return seq[:401]
    elif(length>=501 and length<701):
        return seq[:501]
    elif(length>=701 and length<901):
        return seq[:701]
    elif(length>=901 and length<1001):
        return seq[:901]
    elif(length>=1001):
        return seq[:1001]

# -------------画图区域-----------------------------
# 下面是负责画图的-------------------------------

class SliceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SliceWrapper, self).__init__()
        self.model = model
        # self.target = target

    def forward(self, X):
        # print(self.model(X))
        # print(X.shape)
        out = self.model(X)
        out = out[0][:, 1].unsqueeze(1)
        # print(out.shape)
        result = torch.sigmoid(out)
        #print(result)
        return result


import io


# 对单个序列化成motif的图片
def draw(seq, arr):
    seq = trim_seq(seq, len(seq))
    length = len(seq)

    custom_colors = {
        'A': 'red',
        'C': 'blue',
        'G': 'orange',
        'U': 'green'
    }

    plt.rcParams.update({'font.size': 20})

    if len(arr) > 0:
        index = arr[0]

        x = encode_dna_tensor(seq).to(device)
        x = x.to(torch.int64)
        x = F.one_hot(x, num_classes=4).transpose(1, 2).float()

        model = model_51.to(device)
        pth_path = os.path.join(PARAS_DIR, f"data{index}.pth")
        model.load_state_dict(
            torch.load(pth_path, map_location=device, weights_only=True)['state_dict']
        )

        wrapper = SliceWrapper(model)

        plt.figure(figsize=(10, 5))

        if len(seq) == 51:
            
            total_attr = None
            for _ in range(length - 50):
                X_attr = saturation_mutagenesis(
                    wrapper.cpu(), x.cpu(), start=0, end=50, device=device
                )
                total_attr = X_attr if total_attr is None else total_attr + X_attr

            avg_attr = abs(total_attr)
            ax = plt.subplot(1, 1, 1)
            plot_logo(avg_attr[0, :, :], ax=ax, color=custom_colors)
            ax.set_title('Midpoint of the entire sequence: index=25')

        else:
            print("正常绘制")
            num = (len(seq) - 1) // 100
            for i in range(num):
                X_attr = saturation_mutagenesis(
                    wrapper.cpu(), x.cpu(),
                    start=i * 100, end=(i + 1) * 100, device=device
                )
                avg_attr = abs(X_attr)
                ax = plt.subplot(num, 1, i + 1)
                plot_logo(avg_attr[0, :, :], ax=ax, color=custom_colors)
                ax.set_title(f'Midpoint of the sequence: index={i * 100 + 50}')

        plt.tight_layout()

    else:
        print("不存在")
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.text(
            0.5, 0.5,
            "No relevant modifications detected",
            fontsize=20,
            ha='center', va='center'
        )

    # ====== 统一的 base64 输出 ======
    buf = io.BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64


# -------------预测区域-----------------------------
# 对单一序列进行完整的绘图操作

# 下面是重新进行模型预测的方法，对于一个单一序列进行预测结果的呈现
def predict_item(seq):

    model=model_51
    # 准备输入信息，以及保存原始序列为char_seq, 谁让这个需求是从史山逐步添加的
    char_seq = seq
    length = len(seq)
    seq = encode_dna_tensor(seq)  # 准备模型的输入

    if(length<=50):
        print("序列长度不够")
        return []

    # 信息的存储
    meth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 判断发生了何种甲基化
    index_meth = []  # 发生甲基化的类型
    detail = []  # 三元组[发生甲基化位点, 碱基类型, 发生的修饰类型]
    motif = None

    # 进行多重预测
    for index in range(1, 11):
        print(f"预测内容{index}")
        pth_path = os.path.join(PARAS_DIR, f"data{index}.pth")
        model.load_state_dict(
            torch.load(pth_path, map_location=device, weights_only=True)[
                'state_dict'])  # 加载对应的模型


        model.eval()  # 设为评估模式
        model.to(device)
        seq = seq.to(device)  # 确保输入数据在正确的设备

        # 对第index种类修饰进行预测, 遍历整个序列
        for bit in range(length - 50):
            out, atts, vq_loss, x_recon, perplexity = model(seq[:, bit:bit + 50])
            if out[0, 0].item() < out[0, 1].item():  # 发生了这种甲基化
                if pan(char_seq[bit + 25], index):  # 还要判断一下这种甲基化是否合理
                    detail.append([bit + 25, index, char_seq[bit + 25]])  # 位点，索引，以及实际的字母号
                    if index not in index_meth:
                        index_meth.append(index)  # 如果之前没发生过这种甲基化就要进行一下记录了
                        meth[index - 1] = 1

    # 封装具体的信息
    print("原始的内容为",char_seq)
    print("甲基化类型为",meth)
    print("甲基化种类为",index_meth)
    print("甲基化的具体信息为",detail)

    # 开始进行绘图操作，操作方式如下
    motif_base64=draw(seq,index_meth)

    print("motif图片为",motif_base64[:10])

    # item (生成的item为)



    return {
        'original_seq':char_seq,
        'motif_types':index_meth,
        'details':detail,
        'image':motif_base64
    }




# 用这样的方法可以把模型给处理完
# predict_item(model_51,"AAAAAGUCTCUTCUTCUTCUTCUTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUAUAUAUAUTCAGCATGACCAAUAUAUAUAUAUAUAUAUAUAUAAUCCCCCTCTCTCTCTCTCTCTCTCTCTCCTCTCTCTCTCAAAA")


# 对于多个模型进行绘图的操作



