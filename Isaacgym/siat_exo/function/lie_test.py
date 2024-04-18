import pinocchio as pin
from pinocchio.utils import *
import numpy as np
import math

theta = math.pi/2
c     = math.cos(theta)
s     = math.sin(theta)
R_1 = np.array([[1,0,0],[0,c,-s],[0,s,c]])
p_1 = np.array([0,0,0])


R_2 = eye(3)
p_2 = np.array([0,0,0])
M_target = pin.SE3(R_1,p_1)
M_actual = pin.SE3(R_2,p_2)

E = M_actual.actInv(M_target) 
err = pin.log(E).vector

theta_ln = math.acos((1+c+c-1)/2)
ln_E     = theta_ln/2/math.sin(theta_ln)*np.array([s+s,0,0])
print(ln_E)
print(err[3:])


# A = np.array([1,1,1,1,1])
# b = np.random.rand(5)
# print(A)
# print(b)
# print(A+b)