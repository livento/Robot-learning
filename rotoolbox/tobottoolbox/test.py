import roboticstoolbox as rtb
robot = rtb.models.Panda()
print(robot)
Te = robot.fkine(robot.qr)  # forward kinematics
print(Te)
from spatialmath import SE3

Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tep)         # solve IK
print(sol)
q_pickup = sol[0]
print(robot.fkine(q_pickup))
qt = rtb.jtraj(robot.qr, q_pickup, 50)
robot.plot(qt.q, backend='pyplot', movie='panda1.gif')