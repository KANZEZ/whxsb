import pybullet as p
import pybullet_data
import time
import numpy as np
from pybullet_planning import plan_joint_motion, get_movable_joints, get_joint_positions, set_joint_positions
from model import LPV_NN_3D, single_LPV_3D_vec
import torch
import pandas as pd



# ===== 初始化 pybullet =====
p.connect(p.GUI)  # ✅ 开启可视化窗口
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)  # 设置重力

# ===== 加载环境 =====
block_pos = np.array([2.5, 2.5, 0.5])
robot_base_pos = np.array([3.3, 1.7, 0])
plane_id = p.loadURDF("plane.urdf")
kuka_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True, basePosition=robot_base_pos, baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))
obstacle_id = p.loadURDF("urdf/block2.urdf", useFixedBase=True, basePosition=block_pos)


p.stepSimulation()
time.sleep(2.0)  # 等待加载完成


def get_tangent_basis(r):
    """
    Given r: (3,), unit vector
    Returns: theta_hat, phi_hat (both (3,))
    """

    r = r / np.linalg.norm(r)

    # 选择一个与 r 不平行的向量
    if abs(r[2]) < 0.9:  # r不是接近z轴方向
        v = np.array([0, 0, 1.0])
    else:
        v = np.array([1.0, 0, 0])

    # Gram-Schmidt 正交化
    theta_hat = v - np.dot(v, r) * r
    theta_hat /= np.linalg.norm(theta_hat)

    # 第三个基是 r × theta_hat
    phi_hat = np.cross(r, theta_hat)

    return theta_hat, phi_hat



def get_M(kuka_id, obstacle_id, pos_xyz, block_pos, max_distance=10):
    ####### get sdf values
    closest_points = p.getClosestPoints(
        bodyA=kuka_id,
        bodyB=obstacle_id,
        distance=max_distance
    )
    sdf_values = closest_points[7][8]
    rn = closest_points[7][7]
    e1, e2 = get_tangent_basis(rn)

    A_ref = pos_xyz
    B_ref = block_pos
    Gamma = np.linalg.norm(A_ref - B_ref) / (np.linalg.norm(A_ref - B_ref) - sdf_values)
    lambdan = 1 - 1 / (Gamma)
    lambdae1 = 1 + 1 / (Gamma)
    lambdae2 = 1 + 1 / (Gamma)
    D = np.array([[lambdan, 0, 0], [0, lambdae1, 0], [0, 0, lambdae2]])
    E = np.array([[rn[0], e1[0], e2[0]], [rn[1], e1[1], e2[1]], [rn[2], e1[2], e2[2]]])
    #print(E)

    return E @ D @ np.linalg.inv(E)


def load_model():
    model = LPV_NN_3D(input_dim=3, output_dim=1)
    model.load_state_dict(torch.load('model_weights_3d.pth'))
    model.eval()

    data = pd.read_csv("square_trajectories.csv")
    data = data.to_numpy()
    max_vx = np.max(data[:, 2])
    max_vy = np.max(data[:, 3])
    max_v = min(max_vx, max_vy)

    return model, max_v


################## start the simulation #########################

model, max_v = load_model()


start_pos = np.array([2.8, 1.5, 0.2])
cur_pos = start_pos.copy()

ee_index = 6  # kuka 的末端 link
joint_indices = get_movable_joints(kuka_id)
ik_solution = p.calculateInverseKinematics(
    kuka_id,             # 机器人 ID
    ee_index,            # 末端执行器的 link index
    start_pos
)
ik_solution = ik_solution[:len(joint_indices)]
set_joint_positions(kuka_id, joint_indices, ik_solution)

dt = 0.01
target_end_pos = torch.tensor([4, 3, 0.5], dtype=torch.float32, requires_grad=True)

while True:
    p.stepSimulation()

    input = torch.tensor(cur_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    V = single_LPV_3D_vec(input, target_end_pos, model)
    V_dot = torch.autograd.grad(V, input, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_dot = V_dot.detach().numpy().flatten()
    # get the policy
    vel = np.array([-V_dot[0]/np.linalg.norm(V_dot)*max_v, 
                    -V_dot[1]/np.linalg.norm(V_dot)*max_v,
                    -V_dot[2]/np.linalg.norm(V_dot)])
    ##### policy after modulation:
    modyfied_vel = get_M(kuka_id, obstacle_id, cur_pos, block_pos) @ vel.T

    cur_pos = cur_pos + modyfied_vel * dt

    joint_poses = p.calculateInverseKinematics(kuka_id, ee_index, cur_pos)

    p.setJointMotorControlArray(
        bodyUniqueId=kuka_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=joint_poses
    )

    time.sleep(1/150.0)

