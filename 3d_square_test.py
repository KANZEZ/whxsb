import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import torch
from model import LPV_NN, single_LPV, LPV_NN_3D, single_LPV_3D
import pandas as pd

from modulation import get_M
import map_gen



# create the data points in 3D map
limits = map_gen.limits
z_limits = (0, 1)
x = np.linspace(limits[0], limits[1], 6)
y = np.linspace(limits[2], limits[3], 6)
z = np.linspace(z_limits[0], z_limits[1], 6)
X, Y, Z = np.meshgrid(x, y, z)
pos = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

print(pos.shape)


# load the trained model
model = LPV_NN_3D(input_dim=3, output_dim=1)
model.load_state_dict(torch.load('model_weights_3d.pth'))
model.eval()


# get max velocity:
data = pd.read_csv("square_trajectories_3d.csv")
data = data.to_numpy()
max_vx = np.max(data[:, 3])
max_vy = np.max(data[:, 4])
max_v = min(max_vx, max_vy)


# get the model output (velocity)
vel = np.zeros((pos.shape[0], 3))
#modi_vel = np.zeros((pos.shape[0], 2))
lyap_values = np.zeros((pos.shape[0],))
for i in range(pos.shape[0]):
    input = torch.tensor(pos[i, :], dtype=torch.float32, requires_grad=True).unsqueeze(0)
    # get the Lyapunov value
    V = single_LPV_3D(input, torch.tensor([4.0, 3.0, 0.5], dtype=torch.float32, requires_grad=True), model)
    lyap_values[i] = V.item()
    V_dot = torch.autograd.grad(V, input, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_dot = V_dot.detach().numpy().flatten()
    # get the policy
    vel[i, :] = -V_dot / np.linalg.norm(V_dot) * max_v
    ##### policy after modulation:
    #modyfied_vel = get_M(pos[i, 0], pos[i, 1]) @ vel[i, :].T
    #modi_vel[i, :] = modyfied_vel.T


X = pos[:, 0]
Y = pos[:, 1]
Z = pos[:, 2]
U = vel[:, 0]
V = vel[:, 1]
W = vel[:, 2]
# MU = modi_vel[:, 0]
# MV = modi_vel[:, 1]
speed = np.linalg.norm(vel, axis=1) / 1000
Lyapunov_grid = lyap_values.reshape((len(z), len(y), len(x)))



############# plot map, policy, simulated trajectory #######################
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
# lyapunov
#contour = ax.contourf(x, y, Lyapunov_grid, levels=50, cmap='cividis')
#plt.colorbar(contour, ax=ax, label="Lyapunov Value")

# policy before modulation
ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True, color='red') ### ORIGINAL POLICY
# policy after modulation
#ax.quiver(X, Y, MU, MV, speed, cmap='Reds', scale=20, scale_units='xy') ### MODIFIED POLICY


ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])
ax.set_zlim(z_limits[0], z_limits[1])
#ax.set_box_aspect([1,1,0.5])  # 让xyz比例合理
#ax.set_aspect('equal')
ax.set_title("Draw mouse trajectories, then click 'Store Data'")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.grid(True)

ax.bar3d(
    2, 2, 0,  # 起点 (x, y, z)
    1, 1, 1,  # 尺寸 (dx, dy, dz)
    color='black', alpha=1.0
)
plt.show()