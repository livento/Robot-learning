from __future__ import print_function
import numpy as np
from numpy.linalg import norm, solve
from sys import argv
from os.path import dirname, join, abspath
import os

import pinocchio

def Trans(Q):
    x = Q[0]
    y = Q[1]
    z = Q[2]
    w = Q[3]
    return np.array([[1-2*(y**2+z**2),  2*(x*y-z*w),       2*(x*z+y*w)],
                     [2*(x*y+z*w),      1-2*(x**2+z**2),   2*(y*z-x*w)],
                     [2*(x*z-y*w),      2*(y*z+x*w),       1-2*(x**2+y**2)]])

def Trans_p(p_base,p_link):
    return p_link-p_base

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pinocchio_model_dir = join(DIR, "urdf")
urdf_filename = pinocchio_model_dir + '/SIAT/SIAT.urdf' if len(argv)<2 else argv[1]

model    = pinocchio.buildModelFromUrdf(urdf_filename)

data  = model.createData()

Q = np.array([5.0000906e-01, 5.0000226e-01,  4.9999955e-01, 0.49998942])
r = Trans(Q)
p_base = np.array([-8.9406967e-08,  1.2572855e-08, 1.973])
p_link = np.array([0.20499277, -0.23648897, 1.160998])
p = Trans_p(p_base,p_link)
JOINT_ID = 6
oMdes = pinocchio.SE3(r, p)
  
#q      = pinocchio.neutral(model)
q      = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
eps    = 1e-4
IT_MAX = 1000
DT     = 1e-2
damp   = 1e-12
  
i=0
while True:
    pinocchio.forwardKinematics(model,data,q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)
    err = pinocchio.log(iMd).vector  # in joint frame
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model,data,q,JOINT_ID)  # in joint frame
    J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
    v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model,q,v*DT)
    if not i % 10:
        print('%d: error = %s' % (i, err.T))
    i += 1
  
if success:
    print("Convergence achieved!")
else:
    print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
  
print('\nresult: %s' % q.flatten().tolist())
print('\nfinal error: %s' % err.T)