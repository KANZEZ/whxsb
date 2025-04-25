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


