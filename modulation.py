import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator



############### we need to unify the 2D environment, obstacle shapes later, for now just hard code the shapes to be the same in all the scripts


########## define map and obstacle shape #################
limits = (0, 5, 0, 5)
grid_res=0.01
x_range = np.arange(limits[0], limits[1] + grid_res, grid_res)
y_range = np.arange(limits[2], limits[3] + grid_res, grid_res)
X, Y = np.meshgrid(x_range, y_range)
# Horizontal rectangle: y ∈ [obs_corner[1], obs_corner[1] + obs_width], x ∈ [obs_corner[0], obs_corner[0] + obs_length]

obstacle_mask = np.zeros_like(X, dtype=bool)
obstacle_mask[(Y >= 2.2) & (Y <= 2.2 + 1) &
              (X >= 0.0) & (X <= 4.0)] = True

############# get SDF#######################
outside = distance_transform_edt(~obstacle_mask) * grid_res
inside  = -distance_transform_edt(obstacle_mask) * grid_res
sdf = outside + inside

interp_fn = RegularGridInterpolator((y_range, x_range), sdf, method='linear')

# 查询多个位置（形状为 Nx2）
# query_points = np.array([[1.23, 1.0], [2.5, 2.5], [4, 3.2]]) ############### (y,x)!!!!!!!!!!!
# sdf_values = interp_fn(query_points)
# print(sdf_values)

# get gradient and tangent of the SDF
dy, dx = np.gradient(sdf, grid_res)
norm = np.sqrt(dx**2 + dy**2)
unit_dx = dx / (norm + 1e-8)
unit_dy = dy / (norm + 1e-8)

interp_dx = RegularGridInterpolator((y_range, x_range), dx, method='linear', bounds_error=False, fill_value=np.nan)
interp_dy = RegularGridInterpolator((y_range, x_range), dy, method='linear', bounds_error=False, fill_value=np.nan)
# xy = np.array([[1.23, 1.0], [2.5, 2.5], [4.0, 2.2]])
# yx = xy[:, [1, 0]]
# grad_x = interp_dx(yx)
# grad_y = interp_dy(yx)
# grads = np.stack([grad_x, grad_y], axis=1)  # shape (N, 2)
# print(grads)

dx_flat = unit_dx.ravel()  # shape (N,)
dy_flat = unit_dy.ravel()  # shape (N,)
gradient_vectors = np.stack([dx_flat, dy_flat], axis=1)  # shape (N, 2)

# gradient_vectors: shape (N, 2), 每行是 (dx, dy)
dtx = gradient_vectors[:, 0]
dty = gradient_vectors[:, 1]
tangent_vectors = np.stack([dty, -dtx], axis=1)  # shape (N, 2)


################### produce M(x) matrix for every point in the grid
def get_M(x, y):
    yx = [y,x]
    sdf_values = interp_fn(yx)[0]
    lambdan = 1 - 1 / (1 + sdf_values)
    lambdae = 1 + 1 / (1 + sdf_values)
    D = np.array([[lambdan, 0], [0, lambdae]])

    ######## get gradient
    grad_x = interp_dx(yx)[0]
    grad_y = interp_dy(yx)[0]
    grads = np.array([grad_x, grad_y])
    grads = grads / np.linalg.norm(grads)

    ############# tangent
    e = np.array([grads[1], -grads[0]]) # both directions are ok

    E = np.array([[grads[0], e[0]], [grads[1], e[1]]])

    M = E @ D @ np.linalg.inv(E)

    return M




#################### plot the SDF and gradient
# fig, ax = plt.subplots(figsize=(9, 9))
# plt.contourf(X, Y, sdf, levels=50, cmap='coolwarm')
# plt.colorbar(label='Signed Distance')


# # step = 5
# # plt.quiver(
# #     X[::step, ::step], Y[::step, ::step],
# #     unit_dx[::step, ::step], unit_dy[::step, ::step],
# #     color='black', scale=30
# # )

# ax.set_xlim(limits[0], limits[1])
# ax.set_ylim(limits[2], limits[3])
# ax.set_aspect('equal')
# ax.grid(True)

# obstacle_h = patches.Rectangle((0.0, 2.2), 4.0, 1, facecolor='black')
# ax.add_patch(obstacle_h)
# plt.show()