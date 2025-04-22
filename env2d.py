import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from casadi import MX, vertcat, Function, nlpsol 
import casadi as casa
import cvxpy as cp


class HallMap:
    def __init__(self, map_corner, map_length, map_width, 
                       obs_corner, obs_length, obs_width, grid_res=0.01, if_draw=False):
        self.map_corner = map_corner
        self.map_l = map_length
        self.map_w = map_width
        self.obs_corner = obs_corner
        self.obs_l = obs_length
        self.obs_w = obs_width
        self.grid_res = grid_res

        self.init_sdf(if_draw=if_draw)
        
    ############# not used in optimization #############
    def init_sdf(self, if_draw=False):
        x_range = np.arange(self.map_corner[0], self.map_corner[0] + self.map_l + self.grid_res, self.grid_res)
        y_range = np.arange(self.map_corner[1], self.map_corner[1] + self.map_w + self.grid_res, self.grid_res)
        X, Y = np.meshgrid(x_range, y_range)

        # Define obstacle region (black cross: center + horizontal and vertical lines)
        obstacle_mask = np.zeros_like(X, dtype=bool)

        # Horizontal rectangle:
        # Horizontal rectangle: y ∈ [obs_corner[1], obs_corner[1] + obs_width], x ∈ [obs_corner[0], obs_corner[0] + obs_length]
        obstacle_mask[(Y >= self.obs_corner[1]) & (Y <= self.obs_corner[1] + self.obs_w) &
                      (X >= self.obs_corner[0]) & (X <= self.obs_corner[0] + self.obs_l)] = True

        # Cross rectangle:
        # obstacle_mask[(Y >= 3.0) & (Y <= 4) & (X >= 2) & (X <= 6.0)] = True
        # obstacle_mask[(X >= 3.5) & (X <= 4.5) & (Y >= 2) & (Y <= 5)] = True

        # Calculate signed distance field
        outside_distance = distance_transform_edt(~obstacle_mask) * self.grid_res
        inside_distance = -distance_transform_edt(obstacle_mask) * self.grid_res
        self.sdf_grid = outside_distance + inside_distance
        

        # 创建 SDF 插值器（注意顺序 y, x）
        self.sdf_casadi = casa.interpolant(
            'sdf', 'linear',
            [y_range, x_range],
            self.sdf_grid.ravel(order='F')
        )

        if if_draw: ### draw the sdf map
            plt.figure(figsize=(6, 5))
            plt.contourf(X, Y, self.sdf_grid, levels=50, cmap='coolwarm')
            plt.colorbar(label='Signed Distance')
            plt.contour(X, Y, obstacle_mask, colors='black', linewidths=1)
            plt.title('Signed Distance Field (SDF) of Obstacle')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.show()


    def get_sdf_value(self, point):
        if not hasattr(self, 'sdf_casadi'):
            self.init_sdf()
        return self.sdf_casadi(point)





def traj_opt_using_ipopt():
    ########### map info ############
    map_corner = (0, 0)
    map_length = 8
    map_width = 7
    obs_corner = (0, 3)
    obs_length = 4
    obs_width = 1
    hall_map = HallMap(map_corner, map_length, map_width, obs_corner, obs_length, obs_width, if_draw=True)


    ############### optim param ###########
    ################## IPOPT is very sensitive to the solver params, initial guess, and obstacle size, N, speed...
    N = 50
    dt = 1
    x0 = np.array([1, 1])
    xg = np.array([7, 3])
    d_safe = 0.2
    vlim = (0.5, 0.5)

    x = MX.sym("x", (N+1)*2)
    u = MX.sym("u", N*2)
    z = vertcat(x, u)

    # so x = [xxxxxxxyyyyyyyvxvxvxvxvxvxvyvyvyvyvyvy]
    x_vars = x.reshape(((N+1), 2)) 
    u_vars = u.reshape((N, 2))

    ############### cost function ###########
    cost = 0
    for i in range(N):
        d = x_vars[i + 1, :] - x_vars[i, :]
        cost += (d[0] ** 2 + d[1] ** 2) + 1e-6

    ############### constraint ###########
    g = []
    g += [x_vars[0,0] - x0[0]]
    g += [x_vars[0,1] - x0[1]]
    g += [x_vars[N,0] - xg[0]]
    g += [x_vars[N,1] - xg[1]]

    ###### dynamics #########
    for n in range(N):
        dynx = x_vars[n,0] + dt * u_vars[n,0] - x_vars[n + 1,0]
        dyny = x_vars[n,1] + dt * u_vars[n,1] - x_vars[n + 1,1]
        g += [dynx]
        g += [dyny]

    # # safety
    for i in range(N+1): 
        xi, yi = x_vars[i, 0], x_vars[i, 1]
        sdf_val = hall_map.sdf_casadi(vertcat(yi, xi))  # [y,x]
        g.append(sdf_val - d_safe)
    
    for i in range(N): 
        xi, yi = x_vars[i, 0], x_vars[i, 1]
        xi1, yi1 = x_vars[i+1, 0], x_vars[i+1, 1]
        midx = (xi + xi1) / 2
        midy = (yi + yi1) / 2
        sdf_val = hall_map.sdf_casadi(vertcat(midy, midx))  # [y,x]
        g.append(sdf_val - d_safe)

    # ---------- Step 5: bound ----------
    lbg = np.zeros(len(g))
    ubg = np.zeros(len(g))
    ubg[4+N*2:] = np.inf

    lbx = np.concatenate([np.full(N+1, map_corner[0]),
                          np.full(N+1, map_corner[0]),
                          np.full(N, -vlim[0]),
                          np.full(N, -vlim[0])])
                          
    ubx = np.concatenate([np.full(N+1, map_corner[0] + map_length),
                          np.full(N+1, map_corner[0] + map_width),
                          np.full(N, vlim[0]),
                          np.full(N, vlim[0])])


    x_init = np.linspace(x0, xg, N+1)

    u_init = (xg - x0) / (N * dt)
    u_init = np.tile(u_init, (N, 1))
    z_init = np.concatenate([x_init.flatten(), u_init.flatten()])

    nlp = {'x': z, 'f': cost, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp, {
    'ipopt.tol': 1e-8,
    'ipopt.constr_viol_tol': 1e-8,
    'ipopt.acceptable_tol': 1e-4,
    'ipopt.acceptable_constr_viol_tol': 1e-4,
    'ipopt.max_iter': 2000,
    'ipopt.print_level': 1,
    'print_time': False,
    'ipopt.hessian_approximation': 'limited-memory'
})
    res = solver(x0=z_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    sol = np.array(res['x']).flatten()[:(N+1)*2]
    N = len(sol) // 2
    x_sol = np.stack([sol[:N], sol[N:]], axis=1)
    print("x_sol", x_sol)
    print(x_sol.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.plot(x_sol[:, 0], x_sol[:, 1], 'r-o', label='With Dynamics + speed Limit')
    plt.plot(*x0, 'go', label='Start')
    plt.plot(*xg, 'yo', label='Goal')

    ax.set_xlim(map_corner[0], map_corner[0] + map_length)
    ax.set_ylim(map_corner[1], map_corner[1] + map_width)
    outer_border = patches.Rectangle(map_corner, map_length, map_width, linewidth=10, edgecolor='black', facecolor='white')
    ax.add_patch(outer_border)

    # rectangle obstacle
    obstacle = patches.Rectangle((obs_corner[0], obs_corner[1]), obs_length, obs_width,
                                    linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(obstacle)

    # Cross obstacle
    # horizontal = patches.Rectangle((2, 3), 4, 1, facecolor='black')
    # ax.add_patch(horizontal)
    # vertical = patches.Rectangle((3.5, 2), 1,3 , facecolor='black')
    # ax.add_patch(vertical)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Trajectory Optimization with Dynamics, SDF, and speed Limit")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()




########### hard to solve ###########
def traj_opt_using_MIP():

    ########### map info ############
    map_corner = (0, 0)
    map_length = 8
    map_width = 7
    obs_corner = (0, 3)
    obs_length = 4
    obs_width = 1
    hall_map = HallMap(map_corner, map_length, map_width, obs_corner, obs_length, obs_width, if_draw=True)

    ############### optim param###########
    N = 40
    dt = 1
    x0 = np.array([2.3, 2.])
    xg = np.array([2.5, 6.0])
    d_safe = 0.01
    vlim = (0.5, 0.5) # m/s

    # ---------- var ----------
    x = cp.Variable((N + 1, 2))
    v = cp.Variable((N, 2))  

    #：z_left, z_right, z_down, z_up for each point
    z = cp.Variable((N + 1, 4), boolean=True)
    M = 190  # big-M 

    # ---------- cost ----------
    cost = 0
    for i in range(N):
        cost += cp.square(x[i + 1] - x[i]).sum() + 1e-6

    # ---------- constraints ----------
    constraints = []

    # start-end
    constraints += [x[0] == x0, x[N] == xg]

    # map bounds
    constraints += [x[:, 0] >= map_corner[0], x[:, 0] <= map_corner[0] + map_length]
    constraints += [x[:, 1] >= map_corner[1], x[:, 1] <= map_corner[1] + map_width]
    
    # # dynamics
    for i in range(N):
        constraints += [x[i + 1, 0] - x[i, 0] - dt * v[i, 0] <= 0.1, x[i + 1, 0] - x[i, 0] - dt * v[i, 0] >= -0.1]
        constraints += [x[i + 1, 1] - x[i, 1] - dt * v[i, 1] <= 0.1, x[i + 1, 1] - x[i, 1] - dt * v[i, 1] >= -0.1]

    #speed limit
    for i in range(N):
        constraints += [v[i, 0] <= vlim[0], v[i, 1] <= vlim[1]]
        constraints += [v[i, 0] >= -vlim[0], v[i, 1] >= -vlim[1]]

    #  MIP 
    for i in range(N + 1):
        xi = x[i, 0]
        yi = x[i, 1]

        constraints += [
            xi <= hall_map.obs_corner[0] - d_safe + M * (1 - z[i, 0]),  # left
            xi >= hall_map.obs_corner[0] + hall_map.obs_l + d_safe - M * (1 - z[i, 1]),  # right
            yi <= hall_map.obs_corner[1] - d_safe + M * (1 - z[i, 2]),  # down
            yi >= hall_map.obs_corner[1] + hall_map.obs_w + d_safe - M * (1 - z[i, 3]),  # up
            cp.sum(z[i, :]) >= 1  # OR for four directions
        ]

    # ---------- 求解 ----------
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS_BB, verbose=True)
    x_sol = x.value

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.plot(x_sol[:, 0], x_sol[:, 1], 'r-o', label='With Dynamics + speed Limit')
    plt.plot(*x0, 'go', label='Start')
    plt.plot(*xg, 'yo', label='Goal')

    ax.set_xlim(map_corner[0], map_corner[0] + map_length)
    ax.set_ylim(map_corner[1], map_corner[1] + map_width)

    outer_border = patches.Rectangle(map_corner, map_length, map_width, linewidth=10, edgecolor='black', facecolor='white')
    ax.add_patch(outer_border)

    obstacle = patches.Rectangle((obs_corner[0], obs_corner[1]), obs_length, obs_width,
                                    linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(obstacle)

    plt.gca().set_aspect('equal', adjustable='box')

    
    plt.title("Trajectory Optimization with Dynamics, and speed Limit")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



traj_opt_using_ipopt()
#traj_opt_using_MIP()