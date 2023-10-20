import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

#baseTask定义了机器人强化学习过程中的主要流程。包括创建环境，重置（reset_idx），step，
#获取观察值（reset全部机器人回到初始状态时即为获取观察值时）
class BaseTask():

    def __init__(self, cfg):
        self.gym = gymapi.acquire_gym()