import sys,os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from siat_exo.function.print import print_asset_info,print_actor_info
from siat_exo.function.math import plus
import isaacgym

print(plus(1,2))