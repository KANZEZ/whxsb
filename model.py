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
    norm_diff = torch.norm(x-goal_x)
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

    
class LPV_NN_3D(nn.Module):
    def __init__(self, input_dim, output_dim):
        # (2,128,128,128,1)
        super(LPV_NN_3D, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256,256)
        self.layer3 = nn.Linear(256,256)
        self.layer4 = nn.Linear(256,256)
        self.output = nn.Linear(256, output_dim)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.tanh(self.layer4(x))
        x = self.output(x)
        return x



def single_LPV_3D(x, goal_x, model):
    # x: [1, 3]
    # goal_x: [1, 3]
    diff_phi = model(x-goal_x)
    zero_phi = model(torch.zeros_like(x))
    norm_diff = torch.norm(x-goal_x)
    V = diff_phi - zero_phi + norm_diff
    return V


def loss_fn_3D(x, goal_x, x_dot, model, epsilon, lambda1, lambda2):
    # x: [batch_size, 3]
    # goal_x: [1, 3]
    # x_dot: [batch_size, 3]

    x = x.requires_grad_(True)
    x_size = x.shape[0]
    V = []

    for i in range(x_size):
        V_ = single_LPV_3D(x[i], goal_x, model)
        V.append(V_)

    V = torch.stack(V).reshape(-1).unsqueeze(1)
    

    V_dot = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    loss = torch.tensor([0.0], device=x.device)
    for i in range(x_size-1):
        loss += 1+(torch.matmul(V_dot[i], x_dot[i].T))/(torch.norm(V_dot[i])*torch.norm(x_dot[i]))
        constraint1 = torch.maximum(-V[i], torch.zeros_like(V[i]))
        constraint2 = torch.maximum(V[i+1]-(1-epsilon)*V[i], torch.zeros_like(V[i]))
        loss += lambda1*constraint1.item() + lambda2*constraint2.item()
    return loss



def single_LPV_3D_vec(x, goal_x, model):
    """
    x: [batch_size, 3]
    goal_x: [1, 3]，需要 broadcast
    """
    diff = x - goal_x  # broadcasting (batch_size, 3) - (1, 3)
    diff_phi = model(diff)  # [batch_size, 1]
    zero_phi = model(torch.zeros_like(x))  # [batch_size, 1]
    norm_diff = torch.norm(diff, dim=1, keepdim=True)  # [batch_size, 1]
    
    V = diff_phi - zero_phi + norm_diff  # [batch_size, 1]
    return V


def loss_fn_3D_vec(x, goal_x, x_dot, model, epsilon, lambda1, lambda2):
    """
    x: [batch_size, 3]
    goal_x: [1, 3]
    x_dot: [batch_size, 3]
    """

    x = x.requires_grad_(True)

    # 1. 批量计算 V
    V = single_LPV_3D_vec(x, goal_x, model)  # [batch_size, 1]

    # 2. 批量计算 V_dot
    V_dot = torch.autograd.grad(
        outputs=V,
        inputs=x,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]  # [batch_size, 3]

    # 3. Cosine loss（方向对齐）
    cos_sim = torch.sum(V_dot * x_dot, dim=1) / (torch.norm(V_dot, dim=1) * torch.norm(x_dot, dim=1) + 1e-8)  # [batch_size]
    cos_loss = torch.sum(1 + cos_sim)  # scalar

    # 4. Lyapunov 约束项
    constraint1 = torch.clamp(-V.squeeze(1), min=0.0)  # [batch_size]
    constraint2 = torch.clamp(V[1:].squeeze(1) - (1 - epsilon) * V[:-1].squeeze(1), min=0.0)  # [batch_size-1]

    constraint1_loss = torch.sum(constraint1)
    constraint2_loss = torch.sum(constraint2)

    # 5. 总 loss
    loss = cos_loss + lambda1 * constraint1_loss + lambda2 * constraint2_loss

    return loss
