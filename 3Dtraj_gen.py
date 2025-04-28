import pandas as pd
import numpy as np

traj_2d = pd.read_csv("square_trajectories.csv").to_numpy()
traj_2d_len = pd.read_csv("square_trajectories_len.csv").to_numpy()
traj_2d_cumsum = traj_2d_len.cumsum()

points_idx = 0
traj_idx = 0
total_pts = np.sum(traj_2d_len)

traj_3d = np.array([])
traj_3d_len = []


start_z = np.array([0.2, 0.5, 0.8])
goal_z = 0.5

for i in range(len(traj_2d_len)):
    cur_traj = traj_2d[points_idx:traj_2d_cumsum[i], :]
    z1 = np.ones((cur_traj.shape[0], 1))
    z2 = np.ones((cur_traj.shape[0], 1))
    z3 = np.ones((cur_traj.shape[0], 1))
    vz1 = np.ones((cur_traj.shape[0], 1))
    vz2 = np.ones((cur_traj.shape[0], 1))
    vz3 = np.ones((cur_traj.shape[0], 1))
    for j in range(cur_traj.shape[0]):
        z1[j] = start_z[0]+(goal_z-start_z[0])*(j+1)/cur_traj.shape[0]
        z2[j] = start_z[1]+(goal_z-start_z[1])*(j+1)/cur_traj.shape[0]
        z3[j] = start_z[2]+(goal_z-start_z[2])*(j+1)/cur_traj.shape[0]
        vz1[j] = (goal_z-start_z[0])/cur_traj.shape[0]
        vz2[j] = (goal_z-start_z[1])/cur_traj.shape[0]
        vz3[j] = (goal_z-start_z[2])/cur_traj.shape[0]
    pos_traj1 = np.hstack((cur_traj[:, :2], z1))
    pos_traj2 = np.hstack((cur_traj[:, :2], z2))
    pos_traj3 = np.hstack((cur_traj[:, :2], z3))
    vel_traj1 = np.hstack((cur_traj[:, 2:4], vz1))
    vel_traj2 = np.hstack((cur_traj[:, 2:4], vz2))
    vel_traj3 = np.hstack((cur_traj[:, 2:4], vz3))

    cur_traj_3d = np.hstack((pos_traj1, vel_traj1))
    traj_3d = np.vstack((traj_3d, cur_traj_3d)) if traj_3d.size else cur_traj_3d
    traj_3d_len.append(cur_traj_3d.shape[0])

    cur_traj_3d = np.hstack((pos_traj2, vel_traj2))
    traj_3d = np.vstack((traj_3d, cur_traj_3d)) if traj_3d.size else cur_traj_3d
    traj_3d_len.append(cur_traj_3d.shape[0])

    cur_traj_3d = np.hstack((pos_traj3, vel_traj3))
    traj_3d = np.vstack((traj_3d, cur_traj_3d)) if traj_3d.size else cur_traj_3d
    traj_3d_len.append(cur_traj_3d.shape[0])

    points_idx = traj_2d_cumsum[i]

traj_3d_len = np.array(traj_3d_len).reshape(-1, 1)
traj_3d_cumsum = traj_3d_len.cumsum()

df = pd.DataFrame(traj_3d, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
df.to_csv("square_trajectories_3d.csv", index=False)

df1 = pd.DataFrame(traj_3d_len, columns=["trajectory length"])
df1.to_csv("square_trajectories_3d_len.csv", index=False)

print(traj_2d.shape[0], traj_3d.shape[0], traj_2d_len.shape[0], traj_3d_len.shape[0])

        

