%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise Script for Chapter 3 of:                                       %
% "Robots that can learn and adapt" by Billard, Mirrazavi and Figueroa.   %
% Published Textbook Name: Learning for Adaptive and Reactive Robot       %
% Control, MIT Press 2022                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2020 Learning Algorithms and Systems Laboratory,          %
% EPFL, Switzerland                                                       %
% Author:  Nadia Figueroa                                                 %
% email:   nadia.figueroafernandez@epfl.ch                                %
% website: http://lasa.epfl.ch                                            %
%                                                                         % 
% Modified by Nadia Figueroa on Feb 2025, University of Pennsylvania      %
% email: nadiafig@seas.upenn.edu                                          %
%                                                                         %
% Permission is granted to copy, distribute, and/or modify this program   %
% under the terms of the GNU General Public License, version 2 or any     %
% later version published by the Free Software Foundation.                %
%                                                                         %
% This program is distributed in the hope that it will be useful, but     %
% WITHOUT ANY WARRANTY; without even the implied warranty of              %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General%
% Public License for more details                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 (DATA GENERATION): Draw 2D data with GUI or load dataset %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('ch3_ex3_seDS.m'));
% cd(filepath); %<<== This might be necessary in some machines

% Choose to draw data (true) or load dataset (false)
draw_data = false;

% if draw_data
%     %  Step 1 - OPTION 1 (DATA DRAWING): Draw 2D Dataset with GUI %%
% 
%     run('ch3_ex0_drawData.m');
% 
%     % Shift attractor to origin for SEDS
%     disp('Shifting attractor to origin for SEDS')
%     Data(1:M,:) = Data(1:M,:) - att;
%     Xi_ref = Data(1:M,:);
%     x0_all = x0_all - att;
%     att = [0; 0];
% else
%     %  Step 1 - OPTION 2 (DATA LOADING): Load Motions from LASA Handwriting Dataset %%
%     addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-ds-opt')));
% 
%     % Select one of the motions from the LASA Handwriting Dataset
%     sub_sample      = 2; % Each trajectory has 1000 samples when set to '1'
%     nb_trajectories = 7; % Maximum 7, will select randomly if <7
%     [Data, Data_sh, att, x0_all, ~, dt] = load_LASA_dataset_DS(sub_sample, nb_trajectories);
% 
%     % Position/Velocity Trajectories
%     vel_samples = 15; vel_size = 0.5; 
%     [h_data, ~, ~] = plot_reference_trajectories_DS(Data, att, vel_samples, vel_size);
% 
%     % Extract Position and Velocities
%     M          = size(Data,1) / 2;    
%     Xi_ref     = Data_sh(1:M,:);
%     Xi_dot_ref = Data_sh(M+1:end,:);
% end

% === 1. 读取主数据：x, y, vx, vy ===
raw_data = readmatrix('data/2d-HALL/mouse_trajectories.csv');  % shape: N × 4
% raw_data = readmatrix('mouse_trajectories.csv');  % shape: N × 4
x = raw_data(:,1);
y = raw_data(:,2);
vx = raw_data(:,3);
vy = raw_data(:,4);

% === 2. 读取每段轨迹的长度 ===
traj_lengths = readmatrix('data/2d-HALL/trajectories_len.csv');  % e.g., [274; 224; 159; ...]
% traj_lengths = readmatrix('trajectories_len.csv');  % e.g., [274; 224; 159; ...]
% === 3. 分割轨迹并构造 Data 和 Data_sh ===
M = 2;  % 每个样本维度（位置是2维）
Data = []; Data_sh = []; x0_all = []; x0_end = [];
start_idx = 1;

for i = 1:length(traj_lengths)
    len = traj_lengths(i);
    end_idx = start_idx + len - 1;

    traj = [x(start_idx:end_idx)'; ...
            y(start_idx:end_idx)'; ...
            vx(start_idx:end_idx)'; ...
            vy(start_idx:end_idx)'];

    % 原始轨迹堆叠
    Data = [Data traj];
    x0_all = [x0_all traj(1:2,1)];
    x0_end = [x0_end traj(1:2,end)];

    % 平移每段轨迹的终点到原点
    traj(1:2,:) = traj(1:2,:) - traj(1:2,end);
    traj(3:4,end) = 0;  % 最后一帧速度设为 0
    Data_sh = [Data_sh traj];

    start_idx = end_idx + 1;  % 准备下段轨迹
end

% === 4. 准备数据用于 SEDS ===
Xi_ref = Data_sh(1:M, :);
Xi_dot_ref = Data_sh(M+1:end, :);
att = mean(x0_end, 2);

% === 5. 可选可视化 ===
figure; hold on; axis equal;
plot(Xi_ref(1,:), Xi_ref(2,:), 'b.');
plot(0, 0, 'ro', 'LineWidth', 2);
title('Processed Trajectories for SEDS');

clearvars -except filepath M Xi_ref Xi_dot_ref x0_all Data att draw_data
tStart = cputime;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2a (GMM FITTING): Fit GMM to Trajectory Data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath(fullfile(filepath, '..','..', 'libraries','book-sods-opt')));
addpath(genpath(fullfile(filepath, '..','..', 'libraries','book-phys-gmm')));
addpath(genpath(fullfile(filepath, '..','..', 'libraries','book-thirdparty')));

% 'seds-init': follows the initialization given in the SEDS code
est_options = [];
est_options.type        = 1;   % GMM Estimation Alorithm Type
est_options.maxK        = 10;  % Maximum Gaussians for Type 1/2
est_options.do_plots    = 1;   % Plot Estimation Statistics
est_options.fixed_K     = 5;  % Fix K and estimate with EM
est_options.sub_sample  = 1;   % Size of sub-sampling of trajectories


% 0: Manually set the # of Gaussians
% 1: Do Model Selection with BIC
do_ms_bic = 0;

if do_ms_bic
    [Priors0, ~, ~] = fit_gmm([Xi_ref; Xi_dot_ref], [], est_options);
    nb_gaussians = length(Priors0);
else
    % Select manually the number of Gaussian components
    % Should be at least K=2, so that one is placed on around the attractor
    nb_gaussians = 5;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2b (Optional - Initialize params for SEDS Solver): Initialize GMM parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
init_with_options = 0;

if ~init_with_options
    % Run Algorithm 1 from Chapter 3 (Get an initial guess by deforming Sigma's)
    [Priors0, Mu0, Sigma0] = initialize_SEDS([Xi_ref; Xi_dot_ref], nb_gaussians);
else
    % Run Algorithm 2 from Chapter 3 (Get an initial guess by optimizing
    % each K-th Gaussian function wrt. stability constraints independently
    clear init_options;
    init_options.tol_mat_bias  = 10^-4;
    init_options.tol_stopping  = 10^-10;
    init_options.max_iter      = 500;
    init_options.objective     = 'likelihood';
    [Priors0, Mu0, Sigma0] = initialize_SEDS([Xi_ref; Xi_dot_ref], nb_gaussians, init_options);
end

% Visualize Gaussian Components and labels on clustered trajectories
% Plot Initial Estimate
[~, est_labels] =  my_gmm_cluster([Xi_ref; Xi_dot_ref], Priors0, Mu0, Sigma0, 'hard', []);

% Visualize Estimated Parameters
visualizeEstimatedGMM(Xi_ref, Priors0, Mu0(1:M,:), Sigma0(1:M,1:M,:), est_labels, est_options);
title('GMM $\Theta_{GMR}=\{\pi_k,\mu^k,\Sigma^k\}$ Initial Estimate', 'Interpreter', 'LaTex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 3 (DS ESTIMATION): RUN SEDS SOLVER  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear options;
options.tol_mat_bias  = 10^-4;    % A very small positive scalar to avoid
                                  % instabilities in Gaussian kernel [default: 10^-1]
options.display       = 1;        % An option to control whether the algorithm
                                  % displays the output of each iterations [default: true]
options.tol_stopping  = 10^-10;   % A small positive scalar defining the stoppping
                                  % tolerance for the optimization solver [default: 10^-10]
options.max_iter      = 100;      % Maximum number of iteration for the solver [default: i_max=1000]
options.objective     = 'likelihood';  % 'mse'|'likelihood'
% options.objective     = 'mse';    % 'mse'|'likelihood'
sub_sample            = 1;        % sub-sample trajectories by this factor

% Running SEDS optimization solver
[Priors, Mu, Sigma]= SEDS_Solver(Priors0, Mu0, Sigma0,[Xi_ref(:,1:sub_sample:end); Xi_dot_ref(:,1:sub_sample:end)], options); 
clear ds_seds
ds_seds = @(x) GMR_SEDS(Priors, Mu, Sigma, x - repmat(att,[1 size(x,2)]), 1:M, M+1:2*M);
tEnd = cputime - tStart;

%% %%%%%%%%%%%%    Plot Resulting DS  %%%%%%%%%%%%%%%%%%%
% Fill in plotting options
ds_plot_options = [];
ds_plot_options.sim_traj  = 1;            % To simulate trajectories from x0_all
ds_plot_options.x0_all    = x0_all(:,1:5);       % Intial Points
ds_plot_options.init_type = 'ellipsoid';  % For 3D DS, to initialize streamlines
                                          % 'ellipsoid' or 'cube'
ds_plot_options.nb_points = 30;           % No of streamlines to plot (3D)
ds_plot_options.plot_vol  = 0;            % Plot volume of initial points (3D)

[~, hs, hr, x_sim] = visualizeEstimatedDS(Data(1:M,:), ds_seds, ds_plot_options);
scatter(att(1), att(2), 150, [0 0 0], 'd', 'Linewidth', 2); hold on;
limits = [0, 10, 0, 10];

switch options.objective
    case 'mse'
        title('SEDS $J(\theta_{GMR})$=MSE', 'Interpreter', 'LaTex','FontSize', 30)
    case 'likelihood'
        title('SEDS $J(\theta_{GMR})$=Likelihood', 'Interpreter', 'LaTex', 'FontSize', 30)
end
set(gcf,'Position', [158 389 446 348])

if M == 2
    legend('Dataset trajectories', 'Learned trajectories')
elseif M == 3
    legend('Dataset trajectories', 'Learned DS')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 4 (Optional - Stability Check 2D-only): Plot Lyapunov Function and derivative  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Type of plot
contour = 1; % 0: surf, 1: contour
clear lyap_fun_comb lyap_der
P = eye(2);
% Lyapunov function
lyap_fun = @(x)lyapunov_function_PQLF(x, att, P);
title_string = {'$V(\xi) = (x-x^*)^T(x-x^*)$'};

% Derivative of Lyapunov function (gradV*f(x))
lyap_der = @(x)lyapunov_derivative_PQLF(x, att, P, ds_seds);
title_string_der = {'Lyapunov Function Derivative $\dot{V}(x)$'};

% Plots
h_lyap     = plot_lyap_fct(lyap_fun, contour, limits, title_string, 0);
hd = scatter(Data(1,:), Data(2,:), 10, [1 1 0], 'filled'); hold on;
h_lyap_der = plot_lyap_fct(lyap_der, contour, limits, title_string_der, 1);
hd = scatter(Data(1,:), Data(2,:), 10, [1 1 0], 'filled'); hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 5 (Evaluation): Compute Metrics and Visualize Velocities %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
disp('--------------------')

% Compute RMSE on training data
rmse = mean(rmse_error(ds_seds, Xi_ref, Xi_dot_ref));
fprintf('SEDS got velocity RMSE on training set: %d \n', rmse);

% Compute e_dot on training data
edot = mean(edot_error(ds_seds, Xi_ref, Xi_dot_ref));
fprintf('SEDS got velocity deviation (e_dot) on training set: %d \n', edot);

% Display time 
fprintf('Computation Time is %1.2f seconds \n', tEnd);

% Compute DTWD between train trajectories and reproductions
if ds_plot_options.sim_traj
    nb_traj       = size(x_sim, 3);
    ref_traj_leng = size(Xi_ref, 2) / nb_traj;
    dtwd = zeros(1, nb_traj);
    for n=1:nb_traj
        start_id = round(1 + (n-1) * ref_traj_leng);
        end_id   = round(n * ref_traj_leng);
        dtwd(1,n) = dtw(x_sim(:,:,n)', Data(1:M,start_id:end_id)', 20);
    end
end
fprintf('SEDS got reproduction DTWD on training set: %2.4f +/- %2.4f \n', mean(dtwd), std(dtwd));

% Compare Velocities from Demonstration vs DS
h_vel = visualizeEstimatedVelocities(Data, ds_seds);

%% ===== Step 6: Export SEDS Grid Field and Simulated Trajectory =====

% === Set same limits and resolution as map_gen.py ===
limits = [0, 5, 0, 5];        % xmin, xmax, ymin, ymax
grid_res = 0.01;              % Same resolution

x_vals = limits(1):grid_res:limits(2);
y_vals = limits(3):grid_res:limits(4);
[x_grid, y_grid] = meshgrid(x_vals, y_vals);
grid_points = [x_grid(:)'; y_grid(:)'];

% ===== 推理网格速度和 Lyapunov 值 =====
vel = ds_seds(grid_points);  % 2 x N

P = eye(2);  % Lyapunov 矩阵
V_vals = zeros(1, size(grid_points, 2));
for i = 1:size(grid_points, 2)
    V_vals(i) = lyapunov_function_PQLF(grid_points(:, i), att, P);
end

% ===== 保存网格速度场和 Lyapunov 等高线值 =====
output = [grid_points', vel', V_vals'];
writematrix(output, 'seds_grid_field.csv');

%% ===== Step 7: Simulate One Trajectory from Start to End and Export =====
start_pos = [2.0; 4.0];
end_pos = [1.0; 1.0];
dt = 0.01;
cur_time = 0;
cur_pos = start_pos;
traj_pos = [];
traj_vel = [];
traj_time = [];

max_steps = 5e3;
step = 0;
while norm(cur_pos - end_pos) > 0.005 && step <= max_steps
    v = ds_seds(cur_pos);
    traj_pos(:, end+1) = cur_pos;
    traj_vel(:, end+1) = v;
    traj_time(end+1) = cur_time;
    cur_pos = cur_pos + dt * v;
    cur_time = cur_time + dt;
    step = step + 1;
end

% ===== 导出单条轨迹位置、速度、时间 =====
single_traj = [traj_pos', traj_vel', traj_time'];
writematrix(single_traj, 'seds_single_traj.csv');  % x, y, dx, dy, t

% ===== 导出该轨迹上每一点的 Lyapunov 值 =====
traj_V = zeros(1, size(traj_pos, 2));
for i = 1:size(traj_pos, 2)
    traj_V(i) = lyapunov_function_PQLF(traj_pos(:, i), att, P);
end
writematrix(traj_V', 'seds_single_traj_V.csv');

%% ===== Step 8: Visualize the Single Trajectory =====
figure; hold on; axis equal; grid on;
title('Simulated Single Trajectory under SEDS');
xlabel('x'); ylabel('y');

% ---- 1. Plot Lyapunov contour ----
x_vals = 0:0.05:5;
y_vals = 0:0.05:5;
[x_grid, y_grid] = meshgrid(x_vals, y_vals);
V_plot = zeros(size(x_grid));
P = eye(2);

for i = 1:numel(x_grid)
    x_point = [x_grid(i); y_grid(i)];
    V_plot(i) = lyapunov_function_PQLF(x_point, att, P);
end

contourf(x_grid, y_grid, V_plot, 50, 'LineColor', 'none');
colormap('parula');  % 可换成 turbo/hot
colorbar;
clim([0, max(V_plot(:))]);

% ---- 2. Plot obstacle ----
rectangle('Position', [0.0, 2.2, 4.0, 1.0], 'FaceColor', 'k');
rectangle('Position', [0.0, 1, 4.0, 0.5], 'FaceColor', 'k');

% ---- 3. Plot trajectory ----
plot(traj_pos(1,:), traj_pos(2,:), 'b-', 'LineWidth', 2);
quiver(traj_pos(1,:), traj_pos(2,:), traj_vel(1,:), traj_vel(2,:), ...
       0.5, 'r', 'LineWidth', 1);

legend('Lyapunov V(x)', 'Obstacle', 'Trajectory', 'Velocity');
xlim([0, 5]); ylim([0, 5]);

%% Step 9: Plot Dynamics Field with Single Trajectory
figure; hold on; axis equal; grid on;
title('SEDS Dynamics Field with Single Trajectory');
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 16);

% === 设置坐标轴范围 ===
custom_limits = [0, 5, 0, 5];  % 你可以在这里自定义
xlim(custom_limits(1:2)); ylim(custom_limits(3:4));

% === 1. 绘制黑色流线场 ===
[h_field, ~] = plot_ds_model(gcf, ds_seds, [0; 0], custom_limits, 'medium');  % 200×200 分辨率

% === 2. 计算并绘制轨迹（step7 同样方式） ===
start_pos = [-0.5; 3.76];
end_pos = [1.0; 1.0];
dt = 0.01;
cur_pos = start_pos;
cur_time = 0;
traj_pos = [];
traj_vel = [];
traj_time = [];
Lyv = [];
max_steps = 5e3;
step = 0;

while norm(cur_pos - end_pos) > 0.005 && step <= max_steps
    v = ds_seds(cur_pos);
    traj_pos(:, end+1) = cur_pos;
    traj_vel(:, end+1) = v;
    traj_time(end+1) = cur_time;
    cur_pos = cur_pos + dt * v;
    Lyv(end+1) = lyapunov_function_PQLF(cur_pos, att, P);
    cur_time = cur_time + dt;
    step = step + 1;
end

% === 3. 画蓝色轨迹线 + 黑色箭头表示速度方向 ===
plot(traj_pos(1,:), traj_pos(2,:), 'b-', 'LineWidth', 2);  % 蓝色轨迹线
% quiver(traj_pos(1,:), traj_pos(2,:), traj_vel(1,:), traj_vel(2,:), ...
%        0.5, 'k', 'LineWidth', 1);  % 黑色箭头表示速度

legend({'Vector Field', 'Trajectory', 'Velocity'}, 'Location', 'best');

figure;
plot(traj_time, Lyv);

