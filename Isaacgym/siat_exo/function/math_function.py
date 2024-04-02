import numpy as np

def quaternion2rotation(Q):
    '''
    四元数转化为旋转矩阵
    
    input  : np.array,1*4
 
    output : np.array,3*3
    '''
    x = Q[0]
    y = Q[1]
    z = Q[2]
    w = Q[3]
    return np.array([[1-2*(y**2+z**2),  2*(x*y-z*w),       2*(x*z+y*w)],
                     [2*(x*y+z*w),      1-2*(x**2+z**2),   2*(y*z-x*w)],
                     [2*(x*z-y*w),      2*(y*z+x*w),       1-2*(x**2+y**2)]])

def joint_pr(Base,Link):
    '''
    世界坐标系位姿转为浮动基坐标系位姿
    '''

    state = {'pos':np.zeros(3),
             'rot':np.eye(3)}
    state['pos']= Link['pos']-Base['pos']
    state['rot']= np.dot(quaternion2rotation(Base['rot']),quaternion2rotation(Link['rot']))
    return state