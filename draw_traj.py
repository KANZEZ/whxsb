import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import time

import map_gen

class MouseTrajectoryDrawer:
    def __init__(self, limits=(0, 5, 0, 5)):
        self.limits = limits
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(limits[0], limits[1])
        self.ax.set_ylim(limits[2], limits[3])
        self.ax.set_aspect('equal')
        self.ax.set_title("Draw mouse trajectories, then click 'Store Data'")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)

        # Cross Obs
        self.obstacle_h = map_gen.obstacle_cross[0]
        self.obstacle_v = map_gen.obstacle_cross[1]
        self.ax.add_patch(self.obstacle_h)
        self.ax.add_patch(self.obstacle_v)


        # # Hall Obs
        # self.obstacle_h = map_gen.obstacle_hall
        # self.ax.add_patch(self.obstacle_h)

        # state
        self.drawing = False
        self.X = []  # current trajectory
        self.trials = []  # all trajectories
        self.hp = []  # plot handles

        # UI buttons
        self.store_btn_ax = self.fig.add_axes([0.05, 0.01, 0.15, 0.05])
        self.clear_btn_ax = self.fig.add_axes([0.25, 0.01, 0.15, 0.05])
        self.store_btn = plt.Button(self.store_btn_ax, 'Store Data')
        self.clear_btn = plt.Button(self.clear_btn_ax, 'Clear Data')
        self.store_btn.on_clicked(self.stop_recording)
        self.clear_btn.on_clicked(self.clear_data)

        # connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.button == 1:
            self.drawing = True
            self.X = []
            self.start_time = time.time()

    def on_motion(self, event):
        if self.drawing and event.xdata and event.ydata:
            t = time.time() - self.start_time
            point = [event.xdata, event.ydata, t]
            self.X.append(point)
            dot, = self.ax.plot(event.xdata, event.ydata, 'r.', markersize=4)
            self.hp.append(dot)
            self.fig.canvas.draw()

    def on_release(self, event):
        if event.button == 1:
            self.drawing = False
            if len(self.X) > 2:
                self.trials.append(np.array(self.X).T)  # shape: (3, T)
            print("Stopped trajectory. Total trials:", len(self.trials))

    def stop_recording(self, _):
        print("\nFinalizing and processing trajectories...")
        data_list = []
        traj_len = []
        for traj in self.trials:
            traj_len.append(traj.shape[1])
            pos = traj[:2].T  # shape: (T, 2)
            t = traj[2]
            dt = np.mean(np.diff(t))
            # Savitzky-Golay filter to compute velocities
            pos_smooth = savgol_filter(pos, window_length=15, polyorder=3, axis=0, mode='interp')
            vel = savgol_filter(pos, window_length=15, polyorder=3, deriv=1, delta=dt, axis=0, mode='interp')
            data = np.hstack([pos_smooth, vel])
            data_list.append(data)

        # 输出到 CSV
        all_data = np.vstack(data_list)
        df = pd.DataFrame(all_data, columns=["x", "y", "vx", "vy"])
        df.to_csv("mouse_trajectories.csv", index=False)

        traj_len = np.vstack(traj_len)
        df1 = pd.DataFrame(traj_len, columns=["data"])
        df1.to_csv("trajectories_len.csv", index=False)

        print("Saved to mouse_trajectories.csv")
        print(df.head())

    def clear_data(self, _):
        print("Cleared all data.")
        self.X = []
        self.trials = []
        for handle in self.hp:
            handle.remove()
        self.hp = []
        self.fig.canvas.draw()

    def show(self):
        plt.show()

# 使用示例
drawer = MouseTrajectoryDrawer(limits=(0, 5, 0, 5))
drawer.show()