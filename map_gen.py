import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

############## 2D hall map configuration ###################
limits=(0, 5, 0, 5)
lfx=0.0
lfy=2.2
width=1.0
length=4.0
grid_res=0.01


def get_hall_map():
    """
    Create a hall map with a rectangular obstacle.
    """
    x_range = np.arange(limits[0], limits[1] + grid_res, grid_res)
    y_range = np.arange(limits[2], limits[3] + grid_res, grid_res)
    X, Y = np.meshgrid(x_range, y_range)
    
    obstacle_mask = np.zeros_like(X, dtype=bool)
    obstacle_mask[(Y >= lfy) & (Y <= lfy + width) &
                  (X >= lfx) & (X <= lfx + length)] = True
    
    obstacle = patches.Rectangle((lfx, lfy), length, width, facecolor='black')

    return x_range, y_range, obstacle_mask, obstacle

def get_cross_map():
    """
    Create a hall map with a rectangular obstacle.
    """
    x_range = np.arange(limits[0], limits[1] + grid_res, grid_res)
    y_range = np.arange(limits[2], limits[3] + grid_res, grid_res)
    X, Y = np.meshgrid(x_range, y_range)
    
    obstacle_mask = np.zeros_like(X, dtype=bool)
    obstacle_mask[(Y >= lfy) & (Y <= lfy + width) &
                  (X >= lfx) & (X <= lfx + length)] = True
    
    obstacle = patches.Rectangle((lfx, lfy), length, width, facecolor='black')

    return x_range, y_range, obstacle_mask, obstacle

def get_hall_sdf_info(obstacle_mask, x_range, y_range):
    """
    Get the signed distance function (SDF) of the hall map.
    """
    outside = distance_transform_edt(~obstacle_mask) * grid_res
    inside  = -distance_transform_edt(obstacle_mask) * grid_res
    sdf = outside + inside

    # Interpolate the SDF to get a smooth function
    interp_fn = RegularGridInterpolator((y_range, x_range), sdf, method='linear')

    # get_hall_sdf_gradient
    dy, dx = np.gradient(sdf, grid_res)
    interp_dx = RegularGridInterpolator((y_range, x_range), dx, method='linear', bounds_error=False, fill_value=np.nan)
    interp_dy = RegularGridInterpolator((y_range, x_range), dy, method='linear', bounds_error=False, fill_value=np.nan)

    return sdf, interp_fn, interp_dx, interp_dy


x_range, y_range, obstacle_mask, obstacle = get_hall_map()
sdf, interp_fn, interp_dx, interp_dy = get_hall_sdf_info(obstacle_mask, x_range, y_range)