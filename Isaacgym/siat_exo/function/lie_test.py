import pinocchio as pin
from pinocchio.utils import *
import numpy as np

R_1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
p_1 = np.array([3,3,1])
M_target = pin.SE3(R_1,p_1)

R_2 = eye(3)
p_2 = np.array([0,0,0])
M_actual = pin.SE3(R_2,p_2)

E = M_actual.actInv(M_target) 
err = pin.log(E).vector
print(E)
print(err)
