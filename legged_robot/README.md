# Legged_robot 项目解析和代码笔记
## 一：文件结构
### 1：配置文件
#### 配置文件主要指强化学习训练过程中所需的各项参数。基类位于base_config，主要工程位于legged_robot_config,根据具体项目需重写如cassie_config内容
### 2：任务文件
#### 任务文件指具体强化学习过程，主要包括如何下发角度，如何获取观测值，如何计算奖励，如何判断是否需要重启环境

## 二：未解决问题

### 1： 未解决功能

#### 1：下发角度部分，位控以及力控最终下发均为力矩,下发力矩是由PD控制计算出来的，需要推导。另外action_scale意义不明
        
    def _compute_torques(self,actions):
        """
        actions是PD控制器的位置或速度目标,actions的维度需要与关节数一致
        """

        #pd控制
        #action_scale值为0.5
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

_compute_torques计算结果在step中计算并下发
        
    for _ in range(self.cfg.control.decimation):  #decimation是采样数
        self.torques = self._compute_torques(self.actions).view(self.torques.shape)
        #下发关节扭矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)

### 2： 未解决代码段


### 3： 未解决gym.api


### 4： 仍有疑惑位置