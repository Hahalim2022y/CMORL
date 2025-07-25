from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .zsl1.zsl1_config import ZSL1RoughCfg, ZSL1RoughCfgPPO

import os

from legged_gym.utils.task_registry import task_registry


task_registry.register( "zsl1", LeggedRobot, ZSL1RoughCfg(), ZSL1RoughCfgPPO() )

