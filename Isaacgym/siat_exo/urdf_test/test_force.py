from isaacgym import gymapi,gymutil, gymtorch
import numpy as np
import torch
import random
import os,sys,math


gym = gymapi.acquire_gym()


sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

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

# create sim with these parameters
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "/home/leovento/Robot-learning/urdf"
asset_file = "/box/box.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)
ensor_pose1 = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
box = gym.find_asset_rigid_body_index(asset, "box")
sensor_idx = gym.create_asset_force_sensor(asset, 1, sensor_pose)


spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 8)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)