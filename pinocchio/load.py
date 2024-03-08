import pinocchio
from sys import argv
from os.path import dirname, join, abspath
import os
import numpy as np

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = join(DIR, "urdf")
 
# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = pinocchio_model_dir + '/SIAT/SIAT.urdf' if len(argv)<2 else argv[1]
 
# Load the urdf model
model    = pinocchio.buildModelFromUrdf(urdf_filename)
print('model name: ' + model.name)
 
# Create data required by the algorithms
data     = model.createData()
 
# Sample a random configuration
#q        = pinocchio.randomConfiguration(model)
q        = np.array([9.44236804506783e-06, -1.7675617851919912e-07, -88.21158867581724, 182.73171617851062, -94.52011602834368, 2.839384592443159e-07,0,0,0,0,0,0])
print('q: %s' % q.T)
 
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model,data,q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format( name, *oMi.translation.T.flat )))
