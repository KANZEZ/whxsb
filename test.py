import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import LPV_NN, loss_fn
from read import DataReader
import pandas as pd
# from data import input_data, target_data

datareader = DataReader()
trajectory = datareader.load_2d_data("mouse_trajectories.csv")
traj_len = datareader.load_trajectories_len("trajectories_len.csv")
flag = 0
tra_list = []
for i in range(10):
    tra_list.append(trajectory[flag:flag+traj_len[i,0]])
    flag += traj_len[i,0]


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


# 0.03s = dt
# num_pt = 0
# loss = 0
for i in range(10):
    num_pt = 0
    start_pt = tra_list[i][0, 0:2]
    for j in range(tra_list[i].shape[0]):
        