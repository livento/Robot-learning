from base_config import BaseConfig
from isaacgym import gymapi
import math

class urdfCfg(BaseConfig):
    #定义观测器参数
    class viewer:
        ref_env = 0
        pos = [0, 1, 3]
        lookat = [0, 0, 2.]

    #定义环境参数
    class env:
        num_envs = 16
        envs_per_row =  math.ceil(math.sqrt(num_envs))
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 6.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    #定义asset参数
    class asset:
        asset_root = "/home/leovento/Robot-learning/urdf"
        asset_file = "zongzhuangURDF5/urdf/zongzhuangURDF5.urdf"
        #asset_file = "box/urdf/box.urdf"
        
        #asset_file = "cassie/urdf/cassie.urdf"
        fix_base_link = False
        default_dof_drive_mode = 1

        
    #定义仿真器参数
    class sim_params:
        dt = 1/60
        up_axis = gymapi.UP_AXIS_Z
        gravity = gymapi.Vec3(0,0,-9.8)
        gravity_option = True

    #定义地面参数
    class plane_params:
        normal = gymapi.Vec3(0, 0, 1) # z-up!
        distance = 0            #平面与原点的距离
        static_friction = 1     #静摩擦系数
        dynamic_friction = 1    #动摩擦系数
        restitution = 0         #地面弹性   

    #定义控制器参数
    class control:
        stiffness = {   'l1': 100.0, 'l2': 100.0,
                        'l3': 200., 'l4': 200., 'l5': 200.,
                        'l6': 40.,'r1': 100.0, 'r2': 100.0,
                        'r3': 200., 'r4': 200., 'r5': 200.,
                        'r6': 40.}  # [N*m/rad]
        damping = { 'l1': 3., 'l2': 3.,
                        'l3': 6., 'l4': 6., 'l5': 6.,
                        'l6': 1.,'r1': 3.0, 'r2': 3.0,
                        'r3': 6., 'r4': 6., 'r5': 6.,
                        'r6': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4       
        control_type = 'P'
        
    #定义初始状态
    class init_state:
        init_pos = gymapi.Vec3(0, 0, 0)
        #init_rot = gymapi.Quat(0, 0, 0, 1)
        init_linear_vel = gymapi.Vec3(0, 0, 0)
        init_angular_vel = gymapi.Vec3(0, 0, 0)
        default_joint_angles = {
            "l1":0,
            "l2":0,
            "l3":0,
            "l4":0,
            "l5":0,
            "l6":0,
            "r1":0,
            "r2":0,
            "r3":0,
            "r4":0,
            "r5":0,
            "r6":0
        }