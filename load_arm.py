import pybullet as p
import pybullet_data
import time
import numpy as np
from pybullet_planning import plan_joint_motion, get_movable_joints, get_joint_positions, set_joint_positions


physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.8)

# load ground plane and obstacles
planeId = p.loadURDF("plane.urdf")
table = p.loadURDF("table/table.urdf", basePosition = [1, 0,0])


# load robot arm
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation)

initial_pos = [-0.5, -0.5, 0.7]
initial_ori = p.getQuaternionFromEuler([0, 0, 0])


# kuka 有 7 个关节，从 0 到 6
joints = get_movable_joints(robotId)
print("joints:", joints)
# 获取末端执行器的 link index
ee_index = 6  # kuka 的末端 link


joint_angles = p.calculateInverseKinematics(robotId, ee_index, initial_pos)
print("start planning initial path")
to_initial_path = plan_joint_motion(robotId, joints, joint_angles)
print("finish planning initial path")

if to_initial_path is not None:
    for conf in to_initial_path:
        p.setJointMotorControlArray(
            bodyUniqueId=robotId,
            jointIndices=joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=conf
        )
        p.stepSimulation()
        time.sleep(1. / 60.)
else:
    print("❌ 没找到避障路径，请检查目标是否可达或是否被障碍阻挡")

time.sleep(2)

target_pos = [-0.5, -0.5, 0.4]
target_orn = p.getQuaternionFromEuler([0, 0, 0])
target_joint_angles = p.calculateInverseKinematics(robotId, ee_index, target_pos)


# start_conf = get_joint_positions(robotId, joints)
print("start planning")
path = plan_joint_motion(robotId, joints, target_joint_angles)
np_path = np.array(path)
print("end planning")
print(np_path.shape)
if path is not None:
    for conf in path:
        p.setJointMotorControlArray(
            bodyUniqueId=robotId,
            jointIndices=joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=conf
        )
        p.stepSimulation()
        time.sleep(1. / 240.)
else:
    print("❌ 没找到避障路径，请检查目标是否可达或是否被障碍阻挡")



###################################################################################
# # 起点：末端在桌子上（z ≈ 0.8）
# pos_above = [0, -1, 0.5]
# # 终点：末端到桌子下（z ≈ 0.4）
# pos_below = [0.5, 0.5, 0.5]

# # 使用插值生成一条路径
# num_steps = 1000
# trajectory = np.linspace(pos_above, pos_below, num_steps)

# for target_pos in trajectory:
#     # 求解目标末端位姿的关节角度
#     joint_poses = p.calculateInverseKinematics(robotId, ee_index, target_pos)

#     # 控制机械臂运动到该关节角度
#     p.setJointMotorControlArray(
#         bodyUniqueId=robotId,
#         jointIndices=joint_indices,
#         controlMode=p.POSITION_CONTROL,
#         targetPositions=joint_poses
#     )

#     # 步进模拟
#     p.stepSimulation()
#     time.sleep(1./240.)

try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()