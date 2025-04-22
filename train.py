import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import LPV_NN, loss_fn
# from data import input_data, target_data

batch_size = 64

N = 1000
x = torch.randn(N, 2)
x_next = x + 0.1 * torch.randn(N, 2)
x_dot = x_next - x
xg = torch.tensor([[0.0, 0.0]]) 

train_dataset = TensorDataset(x, x_dot)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = LPV_NN(input_dim=2, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
nepochs = 100

losses = []
for epoch in range(nepochs):
    model.train()
    for x, x_dot in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(x, xg, x_dot, model, epsilon=0.1, lambda1=1.0, lambda2=1.0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# plot losses
plt.plot(range(nepochs),losses, label='loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()



