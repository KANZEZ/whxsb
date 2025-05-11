import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from model import LPV_NN, single_LPV
import pandas as pd

from modulation import get_M
import map_gen

from read import DataReader


# create the data points in 2D map
limits = map_gen.limits
x = np.linspace(limits[0], limits[1], 40)
y = np.linspace(limits[2], limits[3], 40)
X, Y = np.meshgrid(x, y)
pos = np.vstack([X.ravel(), Y.ravel()]).T


# load the trained model
model = LPV_NN(input_dim=2, output_dim=1)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# get max velocity:
data = pd.read_csv("mouse_trajectories.csv")
data = data.to_numpy()
max_vx = np.max(data[:, 2])
max_vy = np.max(data[:, 3])
max_v = min(max_vx, max_vy)


# get the model output (velocity)
vel = np.zeros((pos.shape[0], 2))
modi_vel = np.zeros((pos.shape[0], 2))
lyap_values = np.zeros((pos.shape[0],))
for i in range(pos.shape[0]):
    input = torch.tensor(pos[i, :], dtype=torch.float32, requires_grad=True).unsqueeze(0)
    # get the Lyapunov value
    V = single_LPV(input, torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True), model)
    lyap_values[i] = V.item()
    V_dot = torch.autograd.grad(V, input, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_dot = V_dot.detach().numpy().flatten()
    # get the policy
    vel[i, :] = -V_dot / np.linalg.norm(V_dot) * max_v
    ##### policy after modulation:
    modyfied_vel = get_M(pos[i, 0], pos[i, 1]) @ vel[i, :].T
    modi_vel[i, :] = modyfied_vel.T

################ record the data ####################
X = pos[:, 0].copy()
Y = pos[:, 1].copy()
U = vel[:, 0].copy()
VV = vel[:, 1].copy()
MU = modi_vel[:, 0].copy()
MV = modi_vel[:, 1].copy()
speed = np.linalg.norm(vel, axis=1) / 1000
Lyapunov_grid = lyap_values.reshape((len(y), len(x)))

# ###### simulate the trajectory
# start_pos = np.array([4.2, 4.2])
# end_pos = np.array([1.0, 1.0])
# pos_2d = start_pos
# dt = 0.01
# cur_time = 0
# pos_list, vel_list, time_list, v_list = [], [], [], []

# while np.linalg.norm(pos_2d - end_pos) > 0.005:
    
#     input = torch.tensor(pos_2d, dtype=torch.float32, requires_grad=True).unsqueeze(0)
#     V = single_LPV(input, torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True), model)
#     v_list.append(V.item())
#     V_dot = torch.autograd.grad(V, input, grad_outputs=torch.ones_like(V), create_graph=True)[0]
#     V_dot = V_dot.detach().numpy().flatten()
#     vel_2d = -V_dot / np.linalg.norm(V_dot) * max_v
#     modyfied_vel2d = get_M(pos_2d[0], pos_2d[1]) @ vel_2d.T

#     pos_list.append(pos_2d.copy())  ######### record position, add copy() to get the value not reference 
#     vel_list.append(modyfied_vel2d)
#     time_list.append(cur_time)
#     pos_2d += dt * modyfied_vel2d
#     cur_time += dt


# pos_list = np.array(pos_list)
# vel_list = np.array(vel_list)
# time_list = np.array(time_list)




########## plot all sim traj ###############################
# ###### simulate the trajectory
datareader = DataReader()
trajectory = datareader.load_2d_data("data/2d-HALL/mouse_trajectories.csv")
traj_len = datareader.load_trajectories_len("data/2d-HALL/trajectories_len.csv")
ptr = 0
end_pos = np.array([1.0, 1.0])
big_pos_list = []
total_test_traj = 5
max_iter = 5000
cur_iter = 0

loss_pos = 0
loss_vel = 0
cnt = 0

for i in range(total_test_traj):
    start_pos = trajectory[ptr, 0:2]
    pos_2d = start_pos
    vel_2d = trajectory[ptr, 2:4]
    cur_iter = 0
    ptr += traj_len[i, 0]

    dt = 0.03
    pos_list = [] 
    vel_list = []
    j = 0
    #while np.linalg.norm(pos_2d - end_pos) > 0.01 and cur_iter <= max_iter:
    while j < traj_len[i, 0]:
        loss_pos += np.linalg.norm(pos_2d - trajectory[ptr+j, 0:2])**2
        loss_vel += np.linalg.norm(vel_2d - trajectory[ptr+j, 2:4])**2
        cnt += 1
        j += 1
        cur_iter += 1
        input = torch.tensor(pos_2d, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        V = single_LPV(input, torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True), model)
        V_dot = torch.autograd.grad(V, input, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        V_dot = V_dot.detach().numpy().flatten()
        vel_2d = -V_dot / np.linalg.norm(V_dot) * max_v
        pos_list.append(pos_2d.copy())  ######### record position, add copy() to get the value not reference 
        vel_list.append(vel_2d.copy())
        pos_2d += dt * vel_2d

    if cur_iter != max_iter:
        big_pos_list.append(pos_list)


avg_loss_pos = np.sqrt(loss_pos / cnt)
avg_loss_vel = np.sqrt(loss_vel / cnt)
print(cnt)
print(avg_loss_vel)



# ############## record the position data to csv file ######################
# data = pd.DataFrame(pos_list, columns=['x', 'y'])
# data.to_csv("2dhall_sim_trajdata.csv", index=False)



############# plot map, policy, simulated trajectory #######################
fig, ax = plt.subplots(figsize=(16, 16))
# lyapunov
# contour = ax.contourf(x, y, Lyapunov_grid, levels=50, cmap='cividis')
# plt.colorbar(contour, ax=ax, label="Lyapunov Value")

# policy before modulation
ax.quiver(X, Y, U, VV, speed, cmap='Reds', scale=20, scale_units='xy') ### ORIGINAL POLICY
# policy after modulation
# ax.quiver(X, Y, MU, MV, speed, cmap='Reds', scale=20, scale_units='xy') ### MODIFIED POLICY

############ plot the simulated trajectory
# ax.plot(pos_list[:, 0], pos_list[:, 1], color='blue', linewidth=3, label='Simulated Trajectory')

############### plot all simulated trajectory and dataset trajectory #############
ptr = 0

for i in range(total_test_traj):
    plot_pos_list = []
    for j in range(len(big_pos_list[i])):
        plot_pos_list.append(big_pos_list[i][j])
    plot_pos_list = np.array(plot_pos_list)
    ax.plot(plot_pos_list[:,0], plot_pos_list[:,1], color='blue', linewidth=2, label='Simulated Trajectory')
    ax.scatter(trajectory[ptr:ptr+traj_len[i, 0],0], trajectory[ptr:ptr+traj_len[i, 0],1], color='red', linewidth=2, label='dataset Trajectory')
    ptr += traj_len[i, 0]

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])
ax.set_aspect('equal')
ax.set_title("Draw mouse trajectories, then click 'Store Data'")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# obstacle_h = map_gen.obstacle_hall[0]
# obstacle_v = map_gen.obstacle_cross[1]
# ax.add_patch(obstacle_h)
# ax.add_patch(obstacle_v)

obstacle_h = map_gen.obstacle_hall
ax.add_patch(obstacle_h)
plt.show()



# ################## plot trajectory position velocity #######################
####################### noted that the velocity at the end point is not 0, 
######################### so we only can provide a position reference to a real robot ############
#################### while this learned dynamics is only a guide  for dynamics evolution #################

#fig1, ax1 = plt.subplots()
# # plot trajectory position
# ax1.plot(time_list, v_list, color='blue', linewidth=3, label='Trajectory x position')
# ax1.plot(time_list, pos_list[:, 1], color='yellow', linewidth=3, label='Trajectory y position')
# # plot trajectory velocity
# ax1.plot(time_list, vel_list[:, 0], color='red', linewidth=3, label='Trajectory x velocity')
# ax1.plot(time_list, vel_list[:, 1], color='black', linewidth=3, label='Trajectory y velocity')
# ax1.set_aspect('equal')
# ax1.set_title("Trajectory Position and Velocity")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Position/Velocity")
# ax1.legend()
# ax1.grid(True)
plt.show()