import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

class LPV_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        # (2,128,128,128,1)
        super(LPV_NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,128)
        self.output = nn.Linear(128, output_dim)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.output(x)
        return x
    
def single_LPV(x, goal_x, model):
    # x: [1, 2]
    # goal_x: [1, 2]
    diff_phi = model(x-goal_x)
    zero_phi = model(torch.zeros_like(x))
    norm_diff = torch.norm(x-goal_x, dim=1)
    V = diff_phi - zero_phi + norm_diff
    return V


def loss_fn(x, goal_x, x_dot, model, epsilon, lambda1, lambda2):
    # x: [batch_size, 2]
    # goal_x: [1, 2]
    # x_dot: [batch_size, 2]

    x = x.requires_grad_(True)
    x_size = x.shape[0]
    V = []
    # V_next = []
    for i in range(x_size):
        V_ = single_LPV(x[i], goal_x, model)
        # V_next_ = single_LPV(x[i+1], goal_x, model)
        V.append(V_)
        # V_next.append(V_next_)
        # if i == x_size-2:
        #     V_ = single_LPV(x[i+1], goal_x, model)
        #     V.append(V_)
    V = torch.stack(V).reshape(-1).unsqueeze(1)
    
    # V_next = torch.stack(V_next).reshape(-1)

    V_dot = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    # print(V.shape)
    # print(V_dot.shape)
    loss = torch.tensor([0.0], device=x.device)
    for i in range(x_size-1):
        loss += 1+(torch.matmul(V_dot[i], x_dot[i].T))/(torch.norm(V_dot[i])*torch.norm(x_dot[i]))
        constraint1 = torch.maximum(-V[i], torch.zeros_like(V[i]))
        constraint2 = torch.maximum(V[i+1]-(1-epsilon)*V[i], torch.zeros_like(V[i]))
        loss += lambda1*constraint1.item() + lambda2*constraint2
    return loss

    
