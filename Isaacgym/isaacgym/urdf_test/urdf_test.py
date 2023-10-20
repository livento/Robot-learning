from isaacgym import gymapi
import random
import os
from isaacgym import gymutil
import math
import numpy as np
from function import print_asset_info, print_actor_info


from urdf_config import urdfCfg
from base_task import BaseTask


class urdfTest(BaseTask):
    def __init__(self,cfg:urdfCfg):
        self.cfg = cfg        

        super().__init__(self.cfg)

        self.create_sim()
        self.create_plane()
        self.load_asset()
        self.create_camera()
        self.add_asset()

        self.init_done = True

    def create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.sim_params.dt
        if self.cfg.sim_params.gravity_option:
            sim_params.gravity = self.cfg.sim_params.gravity
        else:
            sim_params.gravity = gymapi.Vec(0,0,0)
        sim_params.up_axis = self.cfg.sim_params.up_axis
        self.sim = self.gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_params)
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
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.fix_base_link = self.cfg.asset.fix_base_link
        self.asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        self.asset = self.gym.load_asset(self.sim, self.cfg.asset.asset_root, self.cfg.asset.asset_file,self.asset_options)
        self.asset_names = ['SIAT']
        self.load_asset_done = True

    def add_asset(self):
        num_envs = self.cfg.env.num_envs
        env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        envs_per_row = self.cfg.env.envs_per_row

        self.envs = []
        self.actor_handles = []

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row) #创建环境
            self.envs.append(env)
            initial_pose = gymapi.Transform()
            initial_pose.p = gymapi.Vec3(0.0, 0, 0)  #每个actor加入时的位置
            #initial_pose.r = gymapi.Quat(-0.707107, 0, 0, 0.707107) #四元组位姿，因为isaacgym是基于y轴向上设计的，因此导入z轴向上的模型时需要进行旋转
            #initial_pose.r = gymapi.Quat(1, 0, 0, 0)
            #为每一个环境添加对象
            actor_handle = self.gym.create_actor(env, self.asset, initial_pose, "MyActor", i, 1)#i指的是碰撞组，只有在统一碰撞组的对象会碰撞，1是位掩码，用于过滤物体碰撞
            self.actor_handles.append(actor_handle)
    
    def play(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            #print_actor_info(gym,envs[0],actor_handles[0])

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)



if __name__ == '__main__':
    cfg = urdfCfg()
    urdf = urdfTest(cfg=cfg)
    urdf.play()