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
import matplotlib.pyplot as plt



def inverse_kinematics(Base,R_foot,L_foot,joint_init=np.zeros(12,dtype=np.float64)):
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
    urdf_filename = pinocchio_model_dir + '/SIAT01/urdf/SIAT01.urdf' if len(argv)<2 else argv[1]
    model    = pinocchio.buildModelFromUrdf(urdf_filename)
    data  = model.createData()

    state_R = joint_pr(Base,R_foot)
    state_L = joint_pr(Base,L_foot)
    
    JOINT_ID_R = 6
    JOINT_ID_L = 12
    
    oMdes_R = pinocchio.SE3(state_R['rot'], state_R['pos'])
    oMdes_L = pinocchio.SE3(state_L['rot'], state_L['pos'])

    #q      = pinocchio.neutral(model)
    
    eps    = 4e-7
    IT_MAX = 200
    DT     = 1    #时间步长
    
    damp   = 1e-6         #DLS的阻尼量
    q      = joint_init

    i=0

    while True:
        #正向
        pinocchio.forwardKinematics(model,data,q)
        #
        iMd_R = data.oMi[JOINT_ID_R].actInv(oMdes_R) 
        iMd_L = data.oMi[JOINT_ID_L].actInv(oMdes_L)
        
        err_R = pinocchio.log(iMd_R).vector  # in joint frame
        err_L = pinocchio.log(iMd_L).vector
        
        if norm(err_R) < eps and norm(err_L) < eps:
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
        q = pinocchio.integrate(model,q,v_R*DT)   #指数映射

        #计算雅可比
        J_L = pinocchio.computeJointJacobian(model,data,q,JOINT_ID_L)  # in joint frame
        J_L = -np.dot(pinocchio.Jlog6(iMd_L.inverse()), J_L)

        #伪逆
        v_L = - J_L.T.dot(solve(J_L.dot(J_L.T) + damp * np.eye(6), err_L))
        q = pinocchio.integrate(model,q,v_L*DT)

        i += 1
            
    if success:
        return q.astype(np.float64)
    else:
        return q.astype(np.float64)
        

if __name__=='__main__':
    import time
    from tqdm import trange
    np.set_printoptions(precision=15)
    trajectory = np.loadtxt('/home/leovento/Robot-learning/Isaacgym/siat_exo/urdf_test/trajectory/trajectory_0413.txt', delimiter=',',dtype=np.float64)
    COM   = trajectory[:,6:9]
    R_pos = trajectory[:,:3]
    L_pos = trajectory[:,3:6]


    # R_base = R_pos - COM
    # L_base = L_pos - COM

    # R_v = np.diff(R_base,axis=0)
    # L_v = np.diff(L_base,axis=0)
    # C_v = np.diff(COM,axis=0)
    # plt.figure()
#对每一列数据绘制曲线图，分别显示在不同的图窗中
#     for i in range(3):
    
#         plt.plot(COM[:, i])
#         plt.title(f'Column {i+1}')
#         plt.xlabel('Index')
#         plt.ylabel('Value')
#         plt.grid(True)


#     plt.figure()
# #对每一列数据绘制曲线图，分别显示在不同的图窗中
#     for i in range(3):
    
#         plt.plot(C_v[:, i])
#         plt.title(f'Column {i+1}')
#         plt.xlabel('Index')
#         plt.ylabel('Value')
#         plt.grid(True)
#     plt.show()
    # data = np.genfromtxt('/home/leovento/Robot-learning/my_gait.txt', delimiter='\t')
    # data_v = np.diff(data)*1000
    # COM_v = np.diff(COM)*1000
    # R_pos_v = np.diff(R_pos)*1000
    # L_pos_v = np.diff(L_pos)*1000
    # num_cols = COM_v.shape[1]

    # plt.figure()
    # plt.plot(data_v)
    # plt.show()
#对每一列数据绘制曲线图，分别显示在不同的图窗中
    # for i in range(num_cols):
    #     plt.figure()
    #     plt.plot(L_pos_v[:, i])
    #     plt.title(f'Column {i+1}')
    #     plt.xlabel('Index')
    #     plt.ylabel('Value')
    #     plt.grid(True)

    # plt.show()

    

    Base =   {'pos':np.array([0,0,0]),
             'rot':np.array([0,0,0,1])}
    R_foot = {'pos':np.array([0,0,0]),
             'rot':np.array([-7.0710993e-01,  0, -7.0710385e-01,  0])}  
    L_foot = {'pos':np.array([0,0,0]),
             'rot':np.array([-7.0710981e-01,  0, -7.0710391e-01,  0])}
    t = 0
    gait = np.zeros(12,dtype=np.float64)
    p_r = np.zeros(12,dtype=np.float64)
    # d   =  np.random.rand(12, 1)/1000
    # start_time = time.time()
    # for t in trange(trajectory.shape[0]):
    # #for t in trange(25000):
    #     if t  == 10000:
    #         print(t)
    #     Base['pos']=COM[t,:]
    #     R_foot['pos'] = R_pos[t,:]
    #     L_foot['pos'] = L_pos[t,:] 
    #     p=inverse_kinematics(Base,R_foot,L_foot,joint_init=p_r)
    #     t=t+1
    #     p_r = p
    #     gait = np.vstack((gait,p))

    for t in trange(trajectory.shape[0]):
        Base['pos']=COM[t,:]
        R_foot['pos'] = R_pos[t,:]
        L_foot['pos'] = L_pos[t,:] 
        p=inverse_kinematics(Base,R_foot,L_foot,joint_init=p_r)
        t=t+1
            #d   =  np.random.rand(12)/1000
        p_r = p
        gait = np.vstack((gait,p))
    np.savetxt('/home/leovento/Robot-learning/Isaacgym/siat_exo/urdf_test/gait/gait_0413.txt', gait, fmt='%.14f')
    # #pos = q.flatten().tolist()
    # #np.savetxt('/home/leovento/Robot-learning/Isaacgym/siat_exo/urdf_test/gait/gait_0402.txt', gait)
