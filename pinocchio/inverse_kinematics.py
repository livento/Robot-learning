#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2024 Inria

"""Example of differential IK stuck in a local minimum.

We set up a simple pendulum with joint limits (0, 2 * pi) radians. That is,
joint angles have to be positive, but the joint can do a full turn. We then
define the goal configuration with an angle of 5.5 rad, and initialize the
robot at joint angle 0.5 rad. We task the pendulum with moving its tip to the
tip-position of the goal configuration.

Because differential IK is a local (think "greedy") algorithm, it will move the
tip on the shortest path to target by turning the pendulum clockwise. This will
locally improve the task error, but eventually the pendulum will just get stuck
at its configuration limit (joint angle = 0 rad). At this stage, differential
IK will just keep pushing against the constraint, which is locally the least
worst thing to do.

The global solution would be to accept a temporary increase in the "current tip
to goal tip" distance, and make the pendulum turn anti-clockwise.
"""

import os

import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask
from pink.visualization import start_meshcat_visualizer

qp_solver = qpsolvers.available_solvers[0]
if "quadprog" in qpsolvers.available_solvers:
    qp_solver = "quadprog"


if __name__ == "__main__":
    # Load robot description from robots/simple_pendulum.urdf
    robot = pin.RobotWrapper.BuildFromURDF(
        filename='/home/leovento/Robot-learning/urdf/SIAT01/urdf/SIAT01.urdf',
        package_dirs=["."],
        root_joint=None,
    )

    # Initialize visualizer and frame task

    model = robot.model
    data = robot.data
    task = FrameTask(
        "tip",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=0.61,  # [cost] / [rad]
    )
    tasks = [task]

    # Feasible goal configuration on one side of configuration limits
    q_goal = np.array([5.5])

    goal_configuration = pink.configuration.Configuration(model, data, q_goal)
    goal_pose = goal_configuration.get_transform_frame_to_world("tip")

    task.set_target_from_configuration(goal_configuration)  # <= task target

    # Initial configuration on the other side of configuration limits
    q_init = np.array([0.5])
    configuration = pink.configuration.Configuration(model, data, q_init)
    init_pose = configuration.get_transform_frame_to_world("tip")

    # Run closed-loop inverse kinematics
    rate = RateLimiter(frequency=100.0)
    dt = rate.period
    while True:
        velocity = solve_ik(configuration, tasks, dt, solver=qp_solver)
        configuration.integrate_inplace(velocity, dt)
        rate.sleep()
