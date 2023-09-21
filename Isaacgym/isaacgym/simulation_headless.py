from isaacgym import gymapi
import random
import os
from isaacgym import gymutil
import math
import numpy as np

#计算下发角度
def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

#输出当前asset状态
def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset) #刚体
    num_joints = gym.get_asset_joint_count(asset)      #关节
    num_dofs = gym.get_asset_dof_count(asset)          #自由度
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):                         
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

#输出当前actor状态
def print_actor_info(gym, env, actor_handle):

    name = gym.get_actor_name(env, actor_handle) #名字

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\n Degrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    # Get body state information
    body_states = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_ALL)

    # Print some state slices
    print("Poses from Body State:")
    print(body_states['pose'])          # print just the poses

    print("\nVelocities from Body State:")
    print(body_states['vel'])          # print just the velocities
    print()

    # iterate through bodies and print name and position
    body_positions = body_states['pose']['p']
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # get DOF states
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # print some state slices
    # Print all states for each degree of freedom
    print(dof_states)
    print()

    # iterate through DOFs and print name and position
    dof_positions = dof_states['pos']
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])

#设置仿真环境属性
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

#设置地面属性
def set_plane(plane_params):
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0            #平面与原点的距离
    plane_params.static_friction = 1     #静摩擦系数
    plane_params.dynamic_friction = 1    #动摩擦系数
    plane_params.restitution = 0         #地面弹性

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
asset_root = "/home/exo/Code/Isaacgym/isaacgym/assets"
asset_file = "USD/cassie/urdf/cassie.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset = gym.load_asset(sim, asset_root, asset_file,asset_options)
asset_names = ['cassie']

#读取模型的自由度信息
dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]


#位控
dof_positions = dof_states['pos']

#驱动器pd以及各关节运动范围限制
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']


#设计各关节运动情况
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        #speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
        speeds[i] = clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        #speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)
        speeds[i] = clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

#创建具体的运行环境
num_envs = 16
envs_per_row = 4
env_spacing = 2.0
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
    initial_pose.p = gymapi.Vec3(0.0, 0, 1.06)  #每个actor加入时的位置

    #initial_pose.r = gymapi.Quat(0, 0, 0, 1)
    #initial_pose.r = gymapi.Quat(-0.707107, 0, 0, 0.707107) #四元组位姿，因为isaacgym是基于y轴向上设计的，因此导入z轴向上的模型时需要进行旋转
    
    #为每一个环境添加对象
    actor_handle = gym.create_actor(env, asset, initial_pose, "MyActor", i, 1)#i指的是碰撞组，只有在统一碰撞组的对象会碰撞，1是位掩码，用于过滤物体碰撞
    actor_handles.append(actor_handle)
    props = gym.get_actor_dof_properties(env, actor_handle)
    #props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
    #props["stffness"] = (5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0)
    #props["damping"] = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    #gym.set_actor_dof_properties(env, actor_handle, props)


#加入相机
cam_props = gymapi.CameraProperties()
cam_pos = gymapi.Vec3(17.2, 0, 4)
cam_target = gymapi.Vec3(5, -0, 3)

viewer = gym.create_viewer(sim, cam_props)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

#gym.write_viewer_image_to_file(viewer.test.png)


#仿真运行参数
dt = sim_params.dt

#运动控制不同阶段命名
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

anim_state = ANIM_SEEK_LOWER
current_dof = 0
show_axis = True

print(lower_limits)
print(upper_limits)



while 1:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    speed = speeds[current_dof]
    if anim_state == ANIM_SEEK_LOWER:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= lower_limits[current_dof]:
            dof_positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_SEEK_UPPER
    elif anim_state == ANIM_SEEK_UPPER:
        dof_positions[current_dof] += speed * dt
        if dof_positions[current_dof] >= upper_limits[current_dof]:
            dof_positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_SEEK_DEFAULT
    if anim_state == ANIM_SEEK_DEFAULT:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= defaults[current_dof]:
            dof_positions[current_dof] = defaults[current_dof]
            anim_state = ANIM_FINISHED
    elif anim_state == ANIM_FINISHED:
        dof_positions[current_dof] = defaults[current_dof]
        current_dof = (current_dof + 1) % num_dofs
        anim_state = ANIM_SEEK_LOWER
        print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)
        if show_axis:
        #get the DOF frame (origin and axis)
            dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
            frame = gym.get_dof_frame(envs[i], dof_handle)

        # draw a line from DOF origin along the DOF axis
            p1 = frame.origin
            p2 = frame.origin + frame.axis * 0.7
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    gym.step_graphics(sim)
    
    #gym.draw_viewer(viewer, sim, True)
    
    gym.sync_frame_time(sim)
    print_actor_info(gym, envs[0], actor_handles[0])

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print(1)