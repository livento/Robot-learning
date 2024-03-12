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
        num_envs = 1
        envs_per_row =  math.ceil(math.sqrt(num_envs))
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 6.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    #定义asset参数
    class asset:
        #asset_root = "/home/leovento/Robot-learning/urdf"
        #asset_file = "zongzhuangURDF5/urdf/zongzhuangURDF5.urdf"
        #asset_file = "zongzhuangURDF5_test/urdf/zongzhuangURDF5.urdf"
        #asset_file = "box/urdf/box.urdf"
        #asset_file = "zongzhuangURDF6/urdf/zongzhuangURDF6.urdf"
        #asset_file = "cassie/urdf/cassie.urdf"
        #asset_file = "lm2/urdf/lm2.urdf"
        asset_file = "SIAT01/urdf/SIAT01.urdf"
        #fix_base_link = False
        fix_base_link = True
        default_dof_drive_mode = 1

        
    #定义仿真器参数
    class sim_params:
        dt = 1/1000
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
        # stiffness = {   'joint_l_1': 100.0, 'joint_l_2': 100.0,
        #                 'joint_l_3': 200., 'joint_l_4': 200., 'joint_l_5': 200.,
        #                 'joint_l_6': 40.,'joint_ar_1': 100.0, 'joint_ar_2': 100.0,
        #                 'joint_ar_3': 200., 'joint_ar_4': 200., 'joint_ar_5': 200.,
        #                 'joint_ar_6': 40.}  # [N*m/rad]
        # damping = { 'joint_l_1': 3., 'joint_l_2': 3.,
        #                 'joint_l_3': 6., 'joint_l_4': 6., 'joint_l_5': 6.,
        #                 'joint_l_6': 1.,'joint_ar_1': 3.0, 'joint_ar_2': 3.0,
        #                 'joint_ar_3': 6., 'joint_ar_4': 6., 'joint_ar_5': 6.,
        #                 'joint_ar_6': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        
        stiffness = {   'joint_l_1': 3000000.0, 'joint_l_2': 3000000.0,
                        'joint_l_3': 3000000., 'joint_l_4': 3000000., 'joint_l_5': 3000000.,
                        'joint_l_6': 3000000.,'joint_ar_1': 3000000.0, 'joint_ar_2': 3000000.0,
                        'joint_ar_3': 3000000., 'joint_ar_4': 3000000., 'joint_ar_5': 3000000.,
                        'joint_ar_6': 3000000.}  # [N*m/rad]
        damping = { 'joint_l_1': 10000., 'joint_l_2': 10000.,
                        'joint_l_3': 10000., 'joint_l_4': 10000., 'joint_l_5': 10000.,
                        'joint_l_6': 10000.,'joint_ar_1': 10000.0, 'joint_ar_2': 10000.0,
                        'joint_ar_3': 10000., 'joint_ar_4': 10000., 'joint_ar_5': 10000.,
                        'joint_ar_6': 10000.} 
        

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4       
        control_type = 'P'
        
    #定义初始状态
    class init_state:
        init_pos = gymapi.Vec3(1, 0, 0)
        #init_rot = gymapi.Quat(0, 0, 0, 1)
        init_linear_vel = gymapi.Vec3(0, 0, 0)
        init_angular_vel = gymapi.Vec3(0, 0, 0)
        default_joint_angles = {
            "joint_l_1":0,
            "joint_l_2":0,
            "joint_l_3":0,
            "joint_l_4":0,
            "joint_l_5":0,
            "joint_l_6":0,
            "joint_ar_1":0,
            "joint_ar_2":0,
            "joint_ar_3":0,
            "joint_ar_4":0,
            "joint_ar_5":0,
            "joint_ar_6":0
        }