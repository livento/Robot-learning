from .base_config import BaseConfig
from isaacgym import gymapi

class urdfCfg(BaseConfig):
    #定义观测器参数
    class viewer:
        ref_env = 0
        pos = [0, 1, 3]
        lookat = [0, 0, 2.]

    #定义环境参数
    class env:
        num_envs = 4096
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class asset:
        file = ""

    class sim_params:
        dt = 1/60
        up_axis = gymapi.UP_AXIS_Z
        gravity = gymapi.Vec(0,0,9.8)
        gravity_option = True

    class plane_params:
        normal = gymapi.Vec3(0, 0, 1) # z-up!
        distance = 0            #平面与原点的距离
        static_friction = 1     #静摩擦系数
        dynamic_friction = 1    #动摩擦系数
        restitution = 0         #地面弹性   