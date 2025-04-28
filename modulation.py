import numpy as np
import map_gen


################### produce M(x) matrix for every point in the grid
def get_M(x, y):
    # ===============================================================
    # hall obstacle
    # yx = [y,x]
    # sdf_values = map_gen.interp_fn_hall(yx)[0]
    # lambdan = 1 - 1 / (1 + sdf_values)
    # lambdae = 1 + 1 / (1 + sdf_values)
    # D = np.array([[lambdan, 0], [0, lambdae]])

    # ######## get gradient
    # grad_x = map_gen.interp_dx_hall(yx)[0]
    # grad_y = map_gen.interp_dy_hall(yx)[0]
    # grads = np.array([grad_x, grad_y])
    # grads = grads / np.linalg.norm(grads)

    # ############# tangent
    # e = np.array([grads[1], -grads[0]]) # both directions are ok

    # E = np.array([[grads[0], e[0]], [grads[1], e[1]]])

    # M = E @ D @ np.linalg.inv(E)

    # ===============================================================
    # cross obstacle
    yx = [y,x]
    sdf_values = map_gen.interp_fn_cross(yx)[0]
    lambdan = 1 - 1 / (1 + sdf_values)
    lambdae = 1 + 1 / (1 + sdf_values)
    D = np.array([[lambdan, 0], [0, lambdae]])

    ######## get gradient
    grad_x = map_gen.interp_dx_cross(yx)[0]
    grad_y = map_gen.interp_dy_cross(yx)[0]
    grads = np.array([grad_x, grad_y])
    grads = grads / np.linalg.norm(grads)

    ############# tangent
    e = np.array([grads[1], -grads[0]]) # both directions are ok

    E = np.array([[grads[0], e[0]], [grads[1], e[1]]])

    M = E @ D @ np.linalg.inv(E)
    return M