from isaacgym import gymapi
import random
import os
from isaacgym import gymutil
import math
import numpy as np

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()

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

# 创建仿真
sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0            #平面与原点的距离
plane_params.static_friction = 1     #静摩擦系数
plane_params.dynamic_friction = 1    #动摩擦系数
plane_params.restitution = 0         #地面弹性

# create the ground plane
gym.add_ground(sim, plane_params)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
cam_pos = gymapi.Vec3(17.2, 0, 4)
cam_target = gymapi.Vec3(5, -0, 3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
