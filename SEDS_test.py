import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import map_gen  # 必须含有 obstacle_cross 和 limits 定义

# ========== Load Data ========== #
grid_data = pd.read_csv("./SEDS/seds_grid_field.csv", header=None).to_numpy()
traj_data = pd.read_csv("./SEDS/seds_single_traj.csv", header=None).to_numpy()
traj_V = pd.read_csv("./SEDS/seds_single_traj_V.csv", header=None).to_numpy()

# ========== Parse Grid Field ========== #
X = grid_data[:, 0]
Y = grid_data[:, 1]
U = grid_data[:, 2]
V = grid_data[:, 3]
Lyapunov = grid_data[:, 4]
speed = np.linalg.norm(np.vstack([U, V]).T, axis=1) / 1000

x_vals = np.unique(X)
y_vals = np.unique(Y)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Lyapunov_grid = Lyapunov.reshape(len(y_vals), len(x_vals))

# ========== Parse Trajectory ========== #
traj_pos = traj_data[:, :2]
traj_vel = traj_data[:, 2:4]
traj_time = traj_data[:, 4]

# ========== Plot: Lyapunov + Vector Field + Trajectory ========== #
fig, ax = plt.subplots(figsize=(16, 16))

# -- Lyapunov contour
contour = ax.contourf(X_grid, Y_grid, Lyapunov_grid, levels=50, cmap='cividis')
plt.colorbar(contour, ax=ax, label="Lyapunov Value")

# -- Velocity field
ax.quiver(X, Y, U, V, speed, cmap='Reds', scale=20, scale_units='xy')

# -- Simulated trajectory
ax.plot(traj_pos[:, 0], traj_pos[:, 1], color='blue', linewidth=3, label='Simulated Trajectory')
ax.quiver(traj_pos[:, 0], traj_pos[:, 1], traj_vel[:, 0], traj_vel[:, 1],
          0.5, color='r', linewidth=1, label='Velocity')

# -- Obstacle from map_gen
obstacle_h = map_gen.obstacle_cross[0]
obstacle_v = map_gen.obstacle_cross[1]
ax.add_patch(obstacle_h)
ax.add_patch(obstacle_v)

# -- Layout
ax.set_xlim(map_gen.limits[0], map_gen.limits[1])
ax.set_ylim(map_gen.limits[2], map_gen.limits[3])
ax.set_aspect('equal')
ax.set_title("Simulated Trajectory under SEDS")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# ========== Plot: Position and Velocity over Time ========== #
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(traj_time, traj_pos[:, 0], label='x position', color='blue', linewidth=2)
ax2.plot(traj_time, traj_pos[:, 1], label='y position', color='orange', linewidth=2)
ax2.plot(traj_time, traj_vel[:, 0], label='x velocity', color='red', linewidth=2)
ax2.plot(traj_time, traj_vel[:, 1], label='y velocity', color='black', linewidth=2)

ax2.set_title("Trajectory Position and Velocity")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Value")
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()
