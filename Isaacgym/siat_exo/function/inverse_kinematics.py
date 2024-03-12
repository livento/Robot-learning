from __future__ import print_function
import numpy as np
from numpy.linalg import norm, solve
from sys import argv
from os.path import dirname, join, abspath
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from siat_exo.function.math_function import joint_pr
import os
import pinocchio

def inverse_kinematics(Base,R_foot,L_foot,joint_init=np.zeros(12,dtype=np.float32)):
    '''
    function:根据指定的Base link位姿以及左右脚的位姿利用逆运动学计算各关节的角度
    Input:  Base:       机器人浮动基的位姿       (numpy.array, dtype=np.float32, shape=7*1)
            R_foot:     右脚位姿                (numpy.array, dtype=np.float32, shape=7*1)
            L_foot:     左脚位姿                (numpy.array, dtype=np.float32, shape=7*1)
            joint_init: 迭代计算的初始点,
                        默认为0,建议为上一时刻角度 (numpy.array, dtype=np.float32, shape=12*1)
    Output: joint_angle:各关节角度               (numpy.array, dtype=np.float32, shape=12*1)
    '''
    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    pinocchio_model_dir = join(DIR, "urdf")
    urdf_filename = pinocchio_model_dir + '/SIAT/SIAT.urdf' if len(argv)<2 else argv[1]
    model    = pinocchio.buildModelFromUrdf(urdf_filename)
    data  = model.createData()

    state_R = joint_pr(Base,R_foot)
    state_L = joint_pr(Base,L_foot)
    
    JOINT_ID_R = 12
    JOINT_ID_L = 6
    
    oMdes_R = pinocchio.SE3(state_R['rot'], state_R['pos'])
    oMdes_L = pinocchio.SE3(state_L['rot'], state_L['pos'])

    #q      = pinocchio.neutral(model)
    q      = joint_init
    eps    = 1e-4
    IT_MAX = 1000
    DT     = 1e-2          #时间步长
    damp   = 1e-12         #求伪逆的阻尼量
    
    i=0
    while True:
        pinocchio.forwardKinematics(model,data,q)
        iMd_R = data.oMi[JOINT_ID_R].actInv(oMdes_R)
        iMd_L = data.oMi[JOINT_ID_L].actInv(oMdes_L)
        err_R = pinocchio.log(iMd_R).vector  # in joint frame
        err_L = pinocchio.log(iMd_L).vector
        err = err_L+err_R
        if norm(err_L) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        #计算雅可比
        J_R = pinocchio.computeJointJacobian(model,data,q,JOINT_ID_R)  # in joint frame
        J_R = -np.dot(pinocchio.Jlog6(iMd_R.inverse()), J_R)

        #伪逆
        v_R = - J_R.T.dot(solve(J_R.dot(J_R.T) + damp * np.eye(6), err_R))
        q = pinocchio.integrate(model,q,v_R*DT)

        #计算雅可比
        J_L = pinocchio.computeJointJacobian(model,data,q,JOINT_ID_L)  # in joint frame
        J_L = -np.dot(pinocchio.Jlog6(iMd_L.inverse()), J_L)

        #伪逆
        v_L = - J_L.T.dot(solve(J_L.dot(J_L.T) + damp * np.eye(6), err_L))
        q = pinocchio.integrate(model,q,v_L*DT).astype(np.float32)

        i += 1
    
    if success:
        return q
    else:
        return joint_init

if __name__=='__main__':
    base =   {'pos':np.array([-8.9406967e-08,  1.2572855e-08, 1.973]),
             'rot':np.array([0,0,0,1])}
    R_foot = {'pos':np.array([2.0499279e-01,  2.3648977e-01, 1.1609977]),
             'rot':np.array([-4.6543719e-06, 7.0711362e-01, -4.0698797e-06, 0.7071001])}
    L_foot = {'pos':np.array([0.20499277, -0.23648897, 1.160998]),
             'rot':np.array([5.0000906e-01, 5.0000226e-01,  4.9999955e-01, 0.49998942])}   
    

    q=inverse_kinematics(base,R_foot,L_foot)
    pos = q.flatten().tolist()
    print('\nresult: %s' % q.flatten().tolist())