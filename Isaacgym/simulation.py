from isaacgym import gymapi
from isaacgym import gymtorch
import math

#Initialize
gym = gymapi.acquire_gym()

#Parameter
sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.physx.use_gpu = True

sim_params.use_gpu_pipeline = True

sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_params) #创建仿真环境

#添加观测器以及修改仿真环境
if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

#导入urdf文件并设置参数
asset_root = "/home/leovento/isaacgym/assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

franka_asset = gym.load_asset(sim,asset_root,franka_asset_file,asset_options)

#创建简单对象(option格式未知)
box_asset = gym.create_box(sim,0.1,0.1,0.1)  

#并行数量
num_envs = 4

#工作空间属性
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
num_per_row = int(math.sqrt(num_envs))

#机械臂姿态
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)   #给每个模型创建工作空间

    franka = gym.create_actor(env,franka_asset,pose,"franka")

gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)