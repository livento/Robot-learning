import numpy as np

def Trans(Q):
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

def joint_pr(p_base,p_link,r_base,r_link):
    '''
    世界坐标系位姿转为joint坐标系位姿
    '''
    return p_link-p_base,np.dot(r_base,r_link)