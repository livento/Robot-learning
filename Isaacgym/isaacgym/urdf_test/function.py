from isaacgym import gymapi
import random
import os
from isaacgym import gymutil
import math
import numpy as np

def print_asset_info(gym,asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset) #刚体
    num_joints = gym.get_asset_joint_count(asset)      #关节
    num_dofs = gym.get_asset_dof_count(asset)          #自由度
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):                         
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

#输出当前actor状态
def print_actor_info(gym, env, actor_handle):
    name = gym.get_actor_name(env, actor_handle) #名字

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\n Degrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    # Get body state information
    body_states = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_ALL)

    # Print some state slices
    print("Poses from Body State:")
    print(body_states['pose'])          # print just the poses

    print("\nVelocities from Body State:")
    print(body_states['vel'])          # print just the velocities
    print()

    # iterate through bodies and print name and position
    body_positions = body_states['pose']['p']
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # get DOF states
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # print some state slices
    # Print all states for each degree of freedom
    print(dof_states)
    print()

    f = open('DOF.txt','w')
    # iterate through DOFs and print name and position
    dof_positions = dof_states['pos']
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])
        for i in range(len(dof_names)):
            f.write(dof_names[i])
            f.write(' ')
            f.write(str(dof_positions[i]))
            f.write('\n')
    
    mass = {
        "base_link":9.51755992218181,
        "R1":1.51779481990376,
        "R2":3.59389626425694,
        "R3":3.65910111444716,
        "R4":2.85321169527239,
        "R5":0.113692025844553,
        "R6":2.40125862032741,
        "L1":1.54071073617692,
        "L2":3.97430335027074,
        "L3":3.65189106558222,
        "L4":3.03366312440795,
        "L5":0.118789875656532,
        "L6":2.4373117048765,
    }
    sum = 0
    mass_sum = 0
    for i in range(len(body_names)):
        sum = 0


if __name__ == "__main__":
    mass = {
        "base_link":9.51755992218181,
        "R1":1.51779481990376,
        "R2":3.59389626425694,
        "R3":3.65910111444716,
        "R4":2.85321169527239,
        "R5":0.113692025844553,
        "R6":2.40125862032741,
        "L1":1.54071073617692,
        "L2":3.97430335027074,
        "L3":3.65189106558222,
        "L4":3.03366312440795,
        "L5":0.118789875656532,
        "L6":2.4373117048765,
    }
 
   