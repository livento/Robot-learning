import numpy as np
import torch
# 读取txt文件数据
data = np.loadtxt('/home/leovento/Robot-learning/Isaacgym/isaacgym/urdf_test/giat/gait/gait.txt', delimiter='\t',dtype=np.float32)
pos = torch.from_numpy(data[0])
print(data[500][2]*10)