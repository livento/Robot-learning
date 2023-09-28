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

#### 2：simparams.dt与self.dt的区别，两者之间相差一个采样项

    def _parse_cfg(self,cfg):
        #时间步为仿真器最小时间单元乘采样率。目前decimation默认值为4
        self.dt = self.cfg.control.decimation * self.sim_params.dt

问题在于暂时不知道两者区别，主要是decimation有什么影响以及sim_params.dt是在什么地方导入默认值

#### 3：step中_post_physics_step_callback中涉及对需要重采样的环境进行重采样的操作
 暂时不懂内部逻辑。该步骤与具体的机器人行走任务有关

    def _post_physics_step_callback(self):
        """ 需要在计算观察值，奖励和结束条件之前调用
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 判断哪些环境需要重采样。当前步长数可被整除的参与重采样？
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
            重采样主要指对当前机器人的速度和方向进行随机化
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

#### 4 init中的_prepare_reward_function涉及奖励值的计算，如何在每一时间步中都进行一次调用？

## 三：目前进度
### 2023.9.28 
笔记 1：实际上在后续调整中，最需要知道的几点分别为：如何获得上一控制周期结束时的机器人各项观测值，如何下发下一时刻的角度，如何自定义奖励计算方法。
其中获取上一时刻观测值和下一时刻下发都是在step中进行。
奖励值的计算主要涉及compute_reward和_prepare_reward_function模块。后者负责计算各项奖励值，前者对每一时间步求和。两者连接通过self.reward_functions进行。如果要自定义奖励函数，重要工作在cfg文件和task中后部分奖励模块处进行，注意需要一一对应。