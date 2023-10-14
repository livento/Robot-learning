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

#### 4 init中的_prepare_reward_function涉及奖励值的计算，如何在每一时间步中都进行一次调用？终止奖励由reset_buf进行传递，终止奖励如何计算？

#### 5 观测值如何计算，为什么需要进行四元数旋转，是否与坐标系有关

    self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)

self.base_lin_vel由quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])得到，其函数功能是实现四元数的旋转

```
    def quat_rotate_inverse(q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c= q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

```

其中
```
self.base_quat=self.root_states[:, 3:7]
self.root_states = gymtorch.wrap_tensor(actor_root_state)
racto_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
```
需要测试self.gym.acquire_actor_root_state_tensor的具体功能，有文档可得

```
Retrieves buffer for Actor root states. The buffer has shape (num_actors, 13). State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).

获取actor root 状态的缓冲区。该缓冲区的形状为（num_actors, 13）。每个演员根的状态包括位置（[0:3]），旋转（[3:7]），线性速度（[7:10]）和角速度（[10:13]）。
```

参考网站
```
https://blog.csdn.net/hongliyu_lvliyu/article/details/127938086
```
中的叙述，新建测试文件。细节工作待整体流程梳理结束后进行。

另一部分中
```
self.obs_scales = self.cfg.normalization.obs_scales

class normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 100.

```
暂不清楚如上系数的功能

#### 6：地面恢复系数（restitution）是什么概念

地面恢复系数指的是地面的弹性系数，介于0到1
。

#### 7:torch_utils中有以下代码功能需要辨别
quat_apply出现于_post_physics_step_callback
```
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)
```

quat_rotate_inverse出现于_init_buffers，用于compute observation
```
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
```

### 2：待学习内容
#### 1：四元数旋转变化等，对应torch.utils中函数
#### 2：三环pid，对应_compute_torques函数
#### 3：gym api
```
gym.get_asset_dof_properties
gym.get_asset_rigid_shape_properties

```



## 三：目前进度
### 2023.9.28 
笔记 1：实际上在后续调整中，最需要知道的几点分别为：如何获得上一控制周期结束时的机器人各项观测值，如何下发下一时刻的角度，如何自定义奖励计算方法。
其中获取上一时刻观测值和下一时刻下发都是在step中进行。
奖励值的计算主要涉及compute_reward和_prepare_reward_function模块。后者负责计算各项奖励值，前者对每一时间步求和。两者连接通过self.reward_functions进行。如果要自定义奖励函数，重要工作在cfg文件和task中后部分奖励模块处进行，注意需要一一对应。

### 2023.10.10
笔记 2：加载模型，加载地面，加载环境

### 2023.10.13
笔记 3：加载asset的时候有参数asset.options.default_dof_drive_mode,该参数决定了关节的具体驱动方式。
```
def _create_envs(self):
    asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
```
一般来讲关节的默认驱动方式应该为力驱，对应到电机上就是电机最终接受的输入是电流信号。这个电流信号由驱动器提供，而驱动器接收的是位控/速控/力控信息，驱动器负责建立位置环/速度环/电流环三环pid生成电流信号发送给电机。对应到仿真器中则是在compute torques中，设定pd控制器的控制方式，p/v/t根据不同模式以及目标值计算下发力矩，实现三环pid的功能。

```
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
```
可以看到在该项目中往往只使用其中一环，只将一项作为追踪目标。另外由位置环计算下发力矩无需动力学信息。

笔记4：在_creat_envs中有关于模型脚的信息，用于加载模型时判断是否与地面产生接触。为了识别哪个刚体属于脚，需要在cfg文件中指明脚的命名
```
feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
```
legged_robot_config中不包括这一信息，默认为NONE，但是在具体的asset的cfg文件中可以指定命名
```
#以下内容位于cassie_config.py
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"
        foot_name = 'toe'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
```