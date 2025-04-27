import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from model import LPV_NN, single_LPV
import pandas as pd

from modulation import get_M
import map_gen


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
X = pos[:, 0]
Y = pos[:, 1]
U = vel[:, 0]
V = vel[:, 1]
MU = modi_vel[:, 0]
MV = modi_vel[:, 1]
speed = np.linalg.norm(vel, axis=1) / 1000
Lyapunov_grid = lyap_values.reshape((len(y), len(x)))




###### simulate the trajectory
start_pos = np.array([2.0, 5.0])
end_pos = np.array([1.0, 1.0])
pos_2d = start_pos
dt = 0.01
cur_time = 0
pos_list, vel_list, time_list = [], [], []

while np.linalg.norm(pos_2d - end_pos) > 0.005:
    input = torch.tensor(pos_2d, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    V = single_LPV(input, torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True), model)
    V_dot = torch.autograd.grad(V, input, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_dot = V_dot.detach().numpy().flatten()
    vel_2d = -V_dot / np.linalg.norm(V_dot) * max_v
    modyfied_vel2d = get_M(pos_2d[0], pos_2d[1]) @ vel_2d.T

    pos_list.append(pos_2d.copy())  ######### record position, add copy() to get the value not reference 
    vel_list.append(modyfied_vel2d)
    time_list.append(cur_time)
    pos_2d += dt * modyfied_vel2d
    cur_time += dt


pos_list = np.array(pos_list)
vel_list = np.array(vel_list)
time_list = np.array(time_list)


############### record the position data to csv file ######################
data = pd.DataFrame(pos_list, columns=['x', 'y'])
data.to_csv("2dhall_sim_trajdata.csv", index=False)



############# plot map, policy, simulated trajectory #######################
fig, ax = plt.subplots(figsize=(16, 16))
# lyapunov
contour = ax.contourf(x, y, Lyapunov_grid, levels=50, cmap='cividis')
plt.colorbar(contour, ax=ax, label="Lyapunov Value")

# policy before modulation
# ax.quiver(X, Y, U, V, speed, cmap='Reds', scale=20, scale_units='xy') ### ORIGINAL POLICY
# policy after modulation
ax.quiver(X, Y, MU, MV, speed, cmap='Reds', scale=20, scale_units='xy') ### MODIFIED POLICY

############ plot the simulated trajectory
ax.plot(pos_list[:, 0], pos_list[:, 1], color='blue', linewidth=3, label='Simulated Trajectory')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])
ax.set_aspect('equal')
ax.set_title("Draw mouse trajectories, then click 'Store Data'")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

obstacle_h = patches.Rectangle((0.0, 2.2), 4.0, 1, facecolor='black')
ax.add_patch(obstacle_h)
plt.show()



# ################## plot trajectory position velocity #######################
####################### noted that the velocity at the end point is not 0, 
######################### so we only can provide a position reference to a real robot ############
#################### while this learned dynamics is only a guide  for dynamics evolution #################

fig1, ax1 = plt.subplots()
# plot trajectory position
ax1.plot(time_list, pos_list[:, 0], color='blue', linewidth=3, label='Trajectory x position')
ax1.plot(time_list, pos_list[:, 1], color='yellow', linewidth=3, label='Trajectory y position')
# plot trajectory velocity
ax1.plot(time_list, vel_list[:, 0], color='red', linewidth=3, label='Trajectory x velocity')
ax1.plot(time_list, vel_list[:, 1], color='black', linewidth=3, label='Trajectory y velocity')
ax1.set_aspect('equal')
ax1.set_title("Trajectory Position and Velocity")
ax1.set_xlabel("Time")
ax1.set_ylabel("Position/Velocity")
ax1.legend()
ax1.grid(True)
plt.show()