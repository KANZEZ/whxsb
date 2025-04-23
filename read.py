import numpy as np
import pandas as pd

class DataReader:
    def __init__(self, bzt = 64):
        self.data = None
        self.traj_len = None
        self.cumulative_traj_len = None
        self.ptr = 0
        self.idx = 0
        self.bzt = bzt

    def load_2d_data(self, filename):
        # Load the data from the CSV file
        self.data = pd.read_csv(filename)
        # Convert to numpy array
        self.data = self.data.to_numpy()
        return self.data

    def load_trajectories_len(self, filename):
        # Load the data from the CSV file
        self.traj_len = pd.read_csv(filename)
        
        # Convert to numpy array
        self.traj_len = self.traj_len.to_numpy()
        self.cumulative_traj_len = self.traj_len.cumsum()

        return self.traj_len

    def get_data(self):
        if self.idx >= len(self.cumulative_traj_len):
            self.idx = 0
            self.ptr = 0

        if self.ptr < self.cumulative_traj_len[self.idx]:
            end_row = min(self.ptr + self.bzt, self.cumulative_traj_len[self.idx])
            x = self.data[self.ptr:end_row, 0:2]
            xd = self.data[self.ptr:end_row, 2:4]
            self.ptr = end_row
            if self.ptr >= self.cumulative_traj_len[self.idx]:
                self.idx += 1
        return x, xd





################# test ########################


# data_reader = DataReader()
# data = data_reader.load_2d_data("mouse_trajectories.csv")
# print("Data shape:", data.shape)
# traj_len = data_reader.load_trajectories_len("trajectories_len.csv")
# print("Trajectory lengths:", traj_len)

# for i in range(6):
#     x, xd = data_reader.get_data()
#     print("x shape:", x.shape)
#     print("xd shape:", xd.shape)
#     print("-----------------")