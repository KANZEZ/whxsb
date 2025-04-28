import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import LPV_NN, loss_fn
from read import DataReader
# from data import input_data, target_data

datareader = DataReader()
trajectory = datareader.load_2d_data("mouse_trajectories.csv")
traj_len = datareader.load_trajectories_len("trajectories_len.csv")

batch_size = 64
total_len = np.sum(traj_len)
traj_sum_len = np.cumsum(traj_len)
traj_goal = np.zeros((traj_len.shape[0], 2))
for i in range(traj_len.shape[0]):
    traj_goal[i, :] = trajectory[traj_sum_len[i]-1, :2]

a = 2

# N = 1000
# x = torch.randn(N, 2)
# x_next = x + 0.1 * torch.randn(N, 2)
# x_dot = x_next - x
# xg = torch.tensor([[0.0, 0.0]]) 

# trajectory_goal = xg

# train_dataset = TensorDataset(x, x_dot)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = LPV_NN(input_dim=2, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
nepochs = 100
plot_cnt = 0
plot_loss = []

losses = []

for epoch in range(nepochs):
    model.train()
    points_cnt = 0
    idx = 0
    epoch_cnt = 0
    total_epoch_loss = 0
    while points_cnt < total_len:
        if points_cnt < traj_sum_len[idx]-1:
            if points_cnt + batch_size <= traj_sum_len[idx]:
                #print(points_cnt)
                
                x = torch.tensor(trajectory[points_cnt:points_cnt+batch_size, :2],dtype=torch.float32)
                x_dot = torch.tensor(trajectory[points_cnt:points_cnt+batch_size, 2:4],dtype=torch.float32)
                xg = torch.tensor(traj_goal[idx, :],dtype=torch.float32)

                points_cnt += batch_size
                optimizer.zero_grad()
                loss = loss_fn(x, xg, x_dot, model, epsilon=0.1, lambda1=1.0, lambda2=1.0)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                total_epoch_loss += loss.item()
                epoch_cnt += 1

            else:
                #print(points_cnt)
                x = torch.tensor(trajectory[points_cnt:traj_sum_len[idx], :2],dtype=torch.float32)
                x_dot = torch.tensor(trajectory[points_cnt:traj_sum_len[idx], 2:4],dtype=torch.float32)
                xg = torch.tensor(traj_goal[idx, :],dtype=torch.float32)
                points_cnt += traj_sum_len[idx] - points_cnt
                optimizer.zero_grad()
                loss = loss_fn(x, xg, x_dot, model, epsilon=0.1, lambda1=1.0, lambda2=1.0)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                total_epoch_loss += loss.item()
                epoch_cnt += 1
        else:
            idx += 1
            # points_cnt += 1

    
    
    plot_cnt += 1
    plot_loss.append(total_epoch_loss / epoch_cnt)
    if epoch % 2 == 0:
        print(f'Epoch {epoch}, Loss: {total_epoch_loss / epoch_cnt}')
        
torch.save(model.state_dict(), 'model_weights.pth')

# plot losses
plt.plot(range(plot_cnt), plot_loss, label='loss', color='blue')
plt.xlabel('Sample Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()



