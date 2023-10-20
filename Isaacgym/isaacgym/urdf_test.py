from isaacgym import gymapi
import random
import os
from isaacgym import gymutil
import math
import numpy as np
from function import print_asset_info, print_actor_info


#设置仿真参数
def set_sim_params(sim_params):
    #sim_params是在配置仿真环境的具体参数
    # set common parameters
    sim_params.dt = 1 / 60
    sim_params.substeps = 2


    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # set Flex-specific parameters
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.8
    sim_params.flex.warm_start = 0.5
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    #sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0)

#设置地面参数
def set_plane(plane_params):
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0            #平面与原点的距离
    plane_params.static_friction = 1     #静摩擦系数
    plane_params.dynamic_friction = 1    #动摩擦系数
    plane_params.restitution = 0         #地面弹性    

#添加URDF路径
asset_root = "/home/leovento/Robot-learning/urdf"
asset_file = "zongzhuangURDF5/urdf/zongzhuangURDF5.urdf"
#asset_file = "lm2/urdf/lm2.urdf"
#asset_file = "cassie/urdf/cassie.urdf"

gym = gymapi.acquire_gym()

#创建仿真环境
sim_params = gymapi.SimParams()
set_sim_params(sim_params)
sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_params)

#创建地面
plane_params = gymapi.PlaneParams()
set_plane(plane_params)
gym.add_ground(sim, plane_params)

#加载模型
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
#asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset = gym.load_asset(sim, asset_root, asset_file,asset_options)
asset_names = ['SIAT']
#print_asset_info(gym,asset, 'SIAT')



#读取模型的自由度信息
dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

#位控
dof_positions = dof_states['pos']

#创建具体的运行环境
num_envs = 16
envs_per_row = 4
env_spacing = 50.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing) #环境大小

envs = []
actor_handles = []

#对每个环境加入actor
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row) #创建环境
    envs.append(env)

    height = random.uniform(1.0, 2.5)

    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0.0, 0, 0)  #每个actor加入时的位置
    #initial_pose.r = gymapi.Quat(-0.707107, 0, 0, 0.707107) #四元组位姿，因为isaacgym是基于y轴向上设计的，因此导入z轴向上的模型时需要进行旋转
    #initial_pose.r = gymapi.Quat(1, 0, 0, 0)
    #为每一个环境添加对象
    actor_handle = gym.create_actor(env, asset, initial_pose, "MyActor", i, 1)#i指的是碰撞组，只有在统一碰撞组的对象会碰撞，1是位掩码，用于过滤物体碰撞
    actor_handles.append(actor_handle)
    props = gym.get_actor_dof_properties(env, actor_handle)


#加入相机
cam_props = gymapi.CameraProperties()
cam_pos = gymapi.Vec3(0, 1, 3)
cam_target = gymapi.Vec3(0, 0, 2)
viewer = gym.create_viewer(sim, cam_props)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    #print_actor_info(gym,envs[0],actor_handles[0])

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print(1)