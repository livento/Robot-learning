import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def gait_interpld(data):
    x_start = np.linspace(2000, 2500, num=11).astype(int)
    y_start = data[x_start]
    x_end = np.linspace(67700, 68200, num=11).astype(int)
    y_end = data[x_end]
    x_new_start = np.linspace(2000, 2500, num=500)
    x_new_end   = np.linspace(67700, 68200, num=500)
    
    for i in range(12):
        f = interp1d(x_start,y_start[:,i].reshape(1,-1),kind='cubic')
        data[2000:2500,i] = f(x_new_start)

        f = interp1d(x_end,y_end[:,i].reshape(1,-1),kind='cubic')
        data[67700:68200,i] = f(x_new_end)
    return data

def gait_interpld_all(data):
    for i in range(132):
        x_start = np.linspace(2000+500*i, 2500+500*i, num=11).astype(int)
        y_start = data[x_start]
    
        x_new_start = np.linspace(2000+500*i, 2500+500*i, num=500)
    
        for j in range(12):
            f = interp1d(x_start,y_start[:,j].reshape(1,-1),kind='cubic')
            data[2000+500*i:2500+500*i,j] = f(x_new_start)

    return data/2/3.1415926*360

if __name__ == '__main__':
    data = data = np.genfromtxt('/home/leovento/Robot-learning/Isaacgym/siat_exo/urdf_test/gait/gait_0402.txt', delimiter=' ') 
    data = gait_interpld(data)
    np.savetxt('/home/leovento/Robot-learning/Isaacgym/siat_exo/urdf_test/gait/gait_0402_inter.txt', data, delimiter='\t',fmt='%.14f')