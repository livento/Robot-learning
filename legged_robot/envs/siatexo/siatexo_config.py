from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SiatexoCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4

        num_observations = 48
        
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.973] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'joint_r_1': 0.,
            'joint_r_2': 0.,
            'joint_r_3': 0.,
            'joint_r_4': 0.,
            'joint_r_5': 0.,
            'joint_r_6': 0.,
            
            'joint_l_1': 0.,
            'joint_l_2': 0.,
            'joint_l_3': 0.,
            'joint_l_4': 0.,
            'joint_l_5': 0.,
            'joint_l_6': 0. 
        }

    class sim(LeggedRobotCfg.sim):
        dt =  0.00025
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'joint_r_1': 100.0, 'joint_r_2': 100.0,
                        'joint_r_3': 200., 'joint_r_4': 200., 'joint_r_5': 200.,
                        'joint_r_6': 40.,
                        'joint_l_1': 100.0, 'joint_l_2': 100.0,
                        'joint_l_3': 200., 'joint_l_4': 200., 'joint_l_5': 200.,
                        'joint_l_6': 40.}  # [N*m/rad]
        
        damping = { 'joint_r_1': 3.0, 'joint_r_2': 3.0,
                    'joint_r_3': 6., 'joint_r_4': 6., 'joint_r_5': 6.,
                    'joint_r_6': 1.,
                    'joint_l_1': 3.0, 'joint_l_2': 3.0,
                    'joint_l_3': 6., 'joint_l_4': 6., 'joint_l_5': 6.,
                    'joint_l_6': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/SIAT/urdf/SIAT.urdf'
        name = "SIAT"
        foot_name = '6'
        terminate_after_contacts_on = ['base_link','joint_r_1','joint_l_1']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 30000.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.

class SiatexoCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'SIATexo'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01