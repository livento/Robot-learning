from isaacgym import gymapi
import random
import os
from isaacgym import gymutil
import math
import numpy as np
from function import print_asset_info, print_actor_info
import gym

from .urdf_config import urdfCfg
from .base_task import BaseTask


class urdfTest(BaseTask):
    def __init__(self,cfg:urdfCfg,physics_engine,sim_device,headless):
        self.cfg = cfg        

        super().__init__(self.cfg, physics_engine, sim_device, headless)

        self.create_sim
        self.create_plane


        if not self.headless:
            self.create_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        self.init_done = True

    def create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.sim_params.dt
        if self.cfg.sim_params.gravity_option:
            sim_params.gravity = self.cfg.sim_params.gravity
        else:
            sim_params.gravity = gymapi.Vec(0,0,0)
        self.sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_params)

    def create_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = self.cfg.plane_params.normal
        plane_params.distance = self.cfg.plane_params.distance   
        plane_params.static_friction = self.cfg.plane_params.static_friction     
        plane_params.dynamic_friction = self.cfg.plane_params.dynamic_friction    
        plane_params.restitution = self.cfg.plane_params.restitution         


    def create_camera(self, position, lookat):
        cam_props = gymapi.CameraProperties()
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        viewer = gym.create_viewer(self.sim, cam_props)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        