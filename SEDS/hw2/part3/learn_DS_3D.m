%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is a template file to import 3D dataset to learn a DS from    %
% trajectories, and exporting the DS to be deployed on the Panda robot.   %
%  
% ====>> You should use functions from part 2 of this homework!           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import dependencies
close all; clear; clc
filepath = fileparts(which('learn_DS_3D.m'));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-ds-opt')));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-sods-opt')));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-phys-gmm')));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-thirdparty')));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-robot-simulation')));
addpath(genpath(fullfile(filepath, 'dataset')));
% cd(filepath); %<<== This might be necessary in some machines

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 (DATA LOADING): Choose among the predifined datasets %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load and convert 3D dataset
% Import from the dataset folder either:
% - Task 1: 'theoretical_DS_dataset.mat'
% - Task 2: 'MPC_train_dataset.mat'
% - Task 2: 'MPC_test_dataset.mat'
% - Task 3: '3D_Cshape_bottom_processed.mat'
% - Task 3: 'raw_demonstration_dataset.mat'

load("MPC_train_dataset.mat"); % --> Modify me to load different datasets!!
% usingSEDS --> Modify me if you will use seds or lpvds! 
% (necessary for parameter storing method used in robot_DS_control.m)
usingSEDS = true;
% filter --> Modify me if you want to pre-process the datasets 
% (relevant for task 3)
filter = false;

% All code below is used to extract trajectories in format amenable to
% learning the DS with the codes provided in part 2 .m scripts
nTraj = size(trajectories, 3);
nPoints = size(trajectories, 2);

Data = [];
attractor = zeros(3, 1);
x0_all = zeros(3, nTraj);

% When filter = true the next lines of code will apply a savitzky golay
% filter to your data (this is recommended for raw human demonstrations)
for i = 1:nTraj
    traj = trajectories(:,:,i);
    if filter
        % Filter Trajectories and Compute Derivativess with Savitzky Golay filter
        %   traj: The trajectory you want to filter
        %   sample_step: subsample the traj before filtering
        %   nth_order :     max order of the derivatives 
        %   n_polynomial :  Order of polynomial fit
        %   window_size :   Window length for the filter
        traj = sgolay_filter_smoothing(trajectories(:,:,i), 5, 1, 2, 10);
    end

    Data = [Data traj];
    x0_all(:,i) = traj(1:3,1);
    attractor = attractor + traj(1:3,end);
end
attractor = attractor / nTraj;

% Normalizing dataset attractor position
M = size(Data, 1) / 2; 
Data(1:M,:) = Data(1:M,:) - attractor;
x0_all = x0_all - attractor;
att = [0; 0; 0];

% Plot position/velocity Trajectories
vel_samples = 5; vel_size = 0.75; 
[h_data, h_att, ~] = plot_reference_trajectories_DS(Data, att, vel_samples, vel_size);

% Extract Position and Velocities
M = size(Data,1) / 2;    
Xi_ref = Data(1:M,:);
Xi_dot_ref  = Data(M+1:end,:);   
axis_limits = axis;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Step 2: ADD YOUR CODE BELOW TO LEARN 3D DS      %%
%% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %%%%%

% 'seds-init': follows the initialization given in the SEDS code
est_options = [];
est_options.type        = 1;   % GMM Estimation Alorithm Type
est_options.maxK        = 10;  % Maximum Gaussians for Type 1/2
est_options.do_plots    = 1;   % Plot Estimation Statistics
est_options.fixed_K     = [];  % Fix K and estimate with EM
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
    nb_gaussians = 2;
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

% Fill in plotting options
ds_plot_options = [];
ds_plot_options.sim_traj = 1; % To simulate trajectories from x0_all
ds_plot_options.x0_all = x0_all; % Iintial Points
ds_plot_options.init_type = 'ellipsoid'; % For 3D DS, to initialize streamlines
% ’ellipsoid’ or ’cube’
ds_plot_options.nb_points = 30; % # of streamlines to plot (3D)
ds_plot_options.plot_vol = 0; % Plot volume of initial points (3D)
[~, hs, hr, x_sim] = visualizeEstimatedDS(Data(1:M,:), ds_seds, ds_plot_options);

clc
disp('--------------------')

% Compute RMSE on training data
rmse = mean(rmse_error(ds_seds, Xi_ref, Xi_dot_ref));
fprintf('SEDS got velocity RMSE on training set: %d \n', rmse);

% Compute e_dot on training data
edot = mean(edot_error(ds_seds, Xi_ref, Xi_dot_ref));
fprintf('SEDS got velocity deviation (e_dot) on training set: %d \n', edot);

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
%%   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ %%%
%%    Step 2: ADD YOUR CODE ABOVE TO LEARN 3D DS      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 3 (SAVE DS): Save learned DS parameters for robot control %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save DS for simulation using 'DS_control.m'
filename = strcat(filepath,'/ds_control.mat');
if usingSEDS
    ds_control = @(x) ds_seds(x - attractor);
    save('ds_control.mat', "ds_control", "attractor", "Priors", "Mu", "Sigma", "att", "M")
else
    ds_control = @(x) ds_lpv(x - attractor);
    save('ds_control.mat', "ds_control", "attractor", "ds_gmm", "A_k", "b_k", "att")
end