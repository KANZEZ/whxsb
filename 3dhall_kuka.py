import pybullet as p
import pybullet_data
import time
import pandas as pd
import numpy as np
from pybullet_planning import plan_joint_motion, get_movable_joints, get_joint_positions, set_joint_positions

############ load the model ##################
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath("/home/hsh/Code/whxsb/urdf/")
table = p.loadURDF("table1.urdf", basePosition = [1, 0, 0])
block = p.loadURDF("block1.urdf", basePosition = [0.5, 0.0, 0.36])
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
# load ground plane and obstacles
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
robotId = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation)
ee_index = 6  # kuka 的末端 link
joint_indices = get_movable_joints(robotId)




################### load the 2D hall sim-trajectory data ##########################

data = pd.read_csv("2dhall_sim_trajdata.csv")
x = data['x'].values
y = data['y'].values
pos = np.column_stack((x, y))
############# since the scale of 2D training env is different from 3D env, we need to scale the data to 3D env
###### length of block is 0.4 in 3D, length of block is 4 in 2D, so the scale is 0.1
scale = 0.1
############ lower left point of block is (0, 2) in 2D, lower left point of block is (0.3, -0.05) in 3D
t = np.array([0.3, -0.05]) - np.array([0, 2]) * scale
pos = pos * scale + t
z = 0.36 *np.ones((pos.shape[0], 1))  # (N, 1) 的一列1
pos_3d = np.hstack((pos, z))    # (N, 3) 的 [x, y, 1]

################## draw the trajectory in 3D ######################
for i in range(len(pos_3d) - 1):
        p.addUserDebugLine(
            pos_3d[i], pos_3d[i+1],
            lineColorRGB=[1, 0, 0],  # 红色
            lineWidth=5.0,
            lifeTime=0   # 0表示永久存在
    )




###################################################################################
############### set the initial position of the end effector
initial_pos = pos_3d[0]
ik_solution = p.calculateInverseKinematics(
    robotId,             # 机器人 ID
    ee_index,            # 末端执行器的 link index
    initial_pos
)

ik_solution = ik_solution[:len(joint_indices)]

set_joint_positions(robotId, joint_indices, ik_solution)


for target_pos in pos_3d:
    # 求解目标末端位姿的关节角度
    joint_poses = p.calculateInverseKinematics(robotId, ee_index, target_pos)

    # 控制机械臂运动到该关节角度
    p.setJointMotorControlArray(
        bodyUniqueId=robotId,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=joint_poses
    )

    # 步进模拟
    p.stepSimulation()
    time.sleep(1./240.)

try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()
