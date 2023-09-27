from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]
    class env:
        num_envs = 4096
    class normalization:
        clip_actions = 100.