from isaacgym import gymapi,gymutil, gymtorch
import numpy as np
import torch
import random
import os,sys,math

from urdf_config import urdfCfg
from base_task import BaseTask

from tqdm import trange

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from siat_exo.function.print import print_asset_info,print_actor_info
from siat_exo.function.inverse_kinematics import inverse_kinematics


class urdfTest(BaseTask):
    def __init__(self,cfg:urdfCfg):
        self.cfg = cfg        

        super().__init__(self.cfg)


        self.create_sim()
        self.create_plane()
        self.load_asset()
        self.create_camera()
        self.add_asset()
        self._init_buffers()
        self.set_actor()
        self.init_done = True

    def create_sim(self):

        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.sim_params.dt
        if self.cfg.sim_params.gravity_option:
            sim_params.gravity = self.cfg.sim_params.gravity
        else:
            sim_params.gravity = gymapi.Vec(0,0,0)
        sim_params.up_axis = self.cfg.sim_params.up_axis
        sim_params.use_gpu_pipeline = False
        sim_device_type = 'cuda'
        self.device = 'cpu'
        self.device_id = 0
        self.sim = self.gym.create_sim(self.device_id,self.device_id,gymapi.SIM_PHYSX, sim_params)
        self.create_sim_done = True

    def create_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = self.cfg.plane_params.normal
        plane_params.distance = self.cfg.plane_params.distance   
        plane_params.static_friction = self.cfg.plane_params.static_friction     
        plane_params.dynamic_friction = self.cfg.plane_params.dynamic_friction    
        plane_params.restitution = self.cfg.plane_params.restitution        
        self.gym.add_ground(self.sim, plane_params)
        self.create_plane_done = True

    def create_camera(self):
        cam_props = gymapi.CameraProperties()
        position = self.cfg.viewer.pos
        lookat   = self.cfg.viewer.lookat
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        self.create_camera_done = True

    def load_asset(self):
        self.asset_root=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),'urdf')
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.fix_base_link = self.cfg.asset.fix_base_link
        self.asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        self.asset = self.gym.load_asset(self.sim, self.asset_root, self.cfg.asset.asset_file,self.asset_options)
        self.asset_names = ['SIAT']
        if self.cfg.asset.add_force_sensor:
            self.add_force()
        self.load_asset_done = True
    def add_asset(self):
        self.num_envs = self.cfg.env.num_envs
        env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        envs_per_row = self.cfg.env.envs_per_row
        self.envs = []
        self.actor_handles = []

        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row) #创建环境
            self.envs.append(env)
            initial_pose = gymapi.Transform()
            initial_pose.p = gymapi.Vec3(0.0, 0, 0.973)              #每个actor加入时的位置
            #initial_pose.r = gymapi.Quat(-0.707107, 0, 0, 0.707107) #四元组位姿，因为isaacgym是基于y轴向上设计的，因此导入z轴向上的模型时需要进行旋转
            #为每一个环境添加对象
            actor_handle = self.gym.create_actor(env, self.asset, initial_pose, "MyActor", i, 1)#i指的是碰撞组，只有在统一碰撞组的对象会碰撞，1是位掩码，用于过滤物体碰撞
    
            self.actor_handles.append(actor_handle)


    def set_actor(self):

        dof_props = self.gym.get_asset_dof_properties(self.asset)
        dof_props["driveMode"][:11].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][:12]=self.p_gains
        dof_props["damping"][:12]=self.d_gains

                
        self.default_dof_state = np.zeros(self.num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        for i in range(self.num_envs):
            self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i],dof_props )

            self.gym.set_actor_dof_states(self.envs[i], self.actor_handles[i], self.default_dof_pos, gymapi.STATE_ALL)

            self.gym.set_actor_dof_position_targets(self.envs[i], self.actor_handles[i], self.default_dof_pos)


    
    def add_force(self):
        right_foot = self.gym.find_asset_rigid_body_index(self.asset, "ar_6")
        left_foot = self.gym.find_asset_rigid_body_index(self.asset, "l_6")
        sensor_pose1 = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
        sensor_pose2 = gymapi.Transform(gymapi.Vec3(-0.2, 0.0, 0.0))

        self.sensor_idx1 = self.gym.create_asset_force_sensor(self.asset, right_foot, sensor_pose1)
        self.sensor_idx2 = self.gym.create_asset_force_sensor(self.asset, right_foot, sensor_pose2)

        # sensor_props = gymapi.ForceSensorProperties()
        # sensor_props.enable_forward_dynamics_forces = True
        # sensor_props.enable_constraint_solver_forces = True
        # sensor_props.use_world_frame = False

        # sensor_idx = self.gym.create_asset_force_sensor(self.asset, right_foot, sensor_pose1, sensor_props)


    def test_dof(self,pos_action):
        '''
        测试给定指定关节角度时的仿真结果
        input : pos_action: np.array,dtype=np.float32
        '''
        pos_action = torch.from_numpy(pos_action)

        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            print_actor_info(self.gym,self.envs[0],self.actor_handles[0])

    def test_gait(self):

        dt = 0
        gait = np.loadtxt('/home/leovento/Robot-learning/Isaacgym/siat_exo/urdf_test/giat/gait.txt', delimiter=' ',dtype=np.float32)
        #gait = gait/360*2*math.pi
        right_positions = []
        left_positions = []
        base_positions = []
        for dt in trange(gait.shape[0]):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            pos_action = torch.from_numpy(gait[dt])
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            body_states = self.gym.get_actor_rigid_body_states(
                self.envs[0], self.actor_handles[0], gymapi.STATE_ALL)
            body_positions = body_states['pose']['p']
            r = body_positions[6].copy()
            l = body_positions[12].copy()
            b = body_positions[0].copy()
            right_positions.append(r)
            left_positions.append(l)
            base_positions.append(b)
        np.savetxt('right_positions.txt', right_positions, fmt='%f')
        np.savetxt('left_positions.txt', left_positions, fmt='%f')
        np.savetxt('base_positions.txt', base_positions, fmt='%f')


    def test_walk(self):  
        trajectory = np.loadtxt('/home/leovento/Robot-learning/Isaacgym/siat_exo/urdf_test/trajectory/trajectory.txt', delimiter=',',dtype=np.float32)
       
        COM   = trajectory[:,6:9]
        R_pos = trajectory[:,:3]
        L_pos = trajectory[:,3:6]
        
        Base =   {'pos':np.array([0,0,0]),
             'rot':np.array([0,0,0,1])}
        R_foot = {'pos':np.array([0,0,0]),
             'rot':np.array([-7.0710993e-01,  0, -7.0710385e-01,  0])}  
        L_foot = {'pos':np.array([0,0,0]),
             'rot':np.array([-7.0710981e-01,  0, -7.0710391e-01,  0])}
        p_r = np.zeros(12,dtype=np.float32)
        dt = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            Base['pos']=COM[dt,:]
            R_foot['pos']=R_pos[dt,:]
            L_foot['pos']=L_pos[dt,:]
            p=inverse_kinematics(Base,R_foot,L_foot,joint_init=p_r)
            p_r = p
            pos_action = torch.from_numpy(p)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            dt = dt+1

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            print(p,dt)
            #print_actor_info(self.gym,self.envs[0],self.actor_handles[0])

    def test_force(self,pos_action):

        pos_action = torch.from_numpy(pos_action)
        if self.cfg.asset.add_force_sensor:
            num_sensors = self.gym.get_actor_force_sensor_count(self.envs[0], self.actor_handles[0])   
            for i in range(num_sensors):
                sensor = self.gym.get_actor_force_sensor(self.envs[0], self.actor_handles[0], i)
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            #刷新机器人状态
            self.gym.refresh_dof_state_tensor(self.sim)

            #设定关节角
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))


            if self.cfg.asset.add_force_sensor:
                sensor_data = sensor.get_forces()

                print(sensor_data.force)   # force as Vec3
                #print(sensor_data.torque) 

            #可视化
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            #print_actor_info(self.gym,self.envs[0],self.actor_handles[0])


    # def test_torque_control(self):
    #     self.default_dof_state = np.zeros(self.num_dofs, gymapi.DofState.dtype)
    #     self.default_dof_state["pos"] = self.default_dof_pos
    #     dof_props = self.gym.get_asset_dof_properties(self.asset)
    #     dof_props["driveMode"][:11].fill(gymapi.DOF_MODE_POS)
    #     gait = np.loadtxt('/home/leovento/Robot-learning/Isaacgym/isaacgym/urdf_test/giat/gait/gait.txt', delimiter='\t',dtype=np.float32)
    #     pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
    #     effort_action = torch.zeros_like(pos_action)

    #     for i in range(self.num_envs):
    #         self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i],dof_props )

    #         self.gym.set_actor_dof_states(self.envs[i], self.actor_handles[i], self.default_dof_pos, gymapi.STATE_ALL)


    #     dt = 0

    #     while not self.gym.query_viewer_has_closed(self.viewer):
    #         # step the physics
    #         self.gym.simulate(self.sim)
    #         self.gym.fetch_results(self.sim, True)
            
    #         self.gym.refresh_dof_state_tensor(self.sim)
            
    #         pos_action = torch.from_numpy(gait[dt])
    #         self.torques = self._compute_torques(pos_action).view(self.torques.shape)
    #         print(self.torques)
    #         self.gym.set./


    #         self.gym.step_graphics(self.sim)
    #         self.gym.draw_viewer(self.viewer, self.sim, True)
    #         self.gym.sync_frame_time(self.sim)


    # def _compute_torques(self, actions):
    #     """ Compute torques from actions.
    #         Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
    #         [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

    #     Args:
    #         actions (torch.Tensor): Actions

    #     Returns:
    #         [torch.Tensor]: Torques sent to the simulation
    #     """
    #     #pd controller
    #     actions_scaled = actions * self.cfg.control.action_scale
    #     control_type = self.cfg.control.control_type
    #     torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

    #     #return torch.clip(torques, -self.torque_limits, self.torque_limits)
    #     return torques

###########################################################################
    def _init_buffers(self):

        
        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        self.dof_names = self.gym.get_asset_dof_names(self.asset)
        #数据先从运行于gpu的gym环境中读取出来再解包成tensors
        # Retrieves buffer for Actor root states. 
        # The buffer has shape (num_actors, 13). 
        # State for each actor root contains position([0:3]),
        # rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # Retrieves Degree-of-Freedom state buffer. 
        # Buffer has shape (num_dofs, 2).
        # Each DOF state contains position and velocity.
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]       
        self.base_quat = self.root_states[:, 3:7]

        

        #Retrieves buffer for net contract forces.
        #The buffer has shape (num_rigid_bodies, 3). 
        #Each contact force state contains one value for each X, Y, Z axis.
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) 

        self.num_actions = self.cfg.env.num_actions
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)     
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)



if __name__ == '__main__':
    cfg = urdfCfg()
    urdf = urdfTest(cfg=cfg)
    
    # base =   {'pos':np.array([0,0,0.973]),
    #          'rot':np.array([0,0,0,1])}
    # R_foot = {'pos':np.array([0.20499277, 0.23648897, 0.133]),
    #          'rot':np.array([5.0000906e-01, 5.0000226e-01,  4.9999955e-01, 0.49998942])}  
    # L_foot = {'pos':np.array([2.0499279e-01,  2.3648977e-01, 1.1609977]),
    #          'rot':np.array([-0, 0.70711362, 0, 0.7071001])}
    # p=inverse_kinematics(base,R_foot,L_foot)
    #p=np.zeros(12,dtype=np.float32)
    #p = np.array([6.983752300256228e-05, 3.6597100161316084e-06, -2.0045930213485716, 3.5132167615390184, -1.5086114712597833, -6.613464983913968e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float32)
    urdf.test_gait()
    #urdf.test_stand()
    #urdf.test_dof(pos_action=p)
    #urdf.test_force(p)
    #urdf.test_walk()