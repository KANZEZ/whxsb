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

load("raw_demonstration_dataset.mat"); % --> Modify me to load different datasets!!
% usingSEDS --> Modify me if you will use seds or lpvds! 
% (necessary for parameter storing method used in robot_DS_control.m)
usingSEDS = false;
% filter --> Modify me if you want to pre-process the datasets 
% (relevant for task 3)
filter = true;

% All code below is used to extract trajectories in format amenable to
% learning the DS with the codes provided in part 2 .m scripts
nTraj = size(trajectories, 3);
nPoints = size(trajectories, 2);

Data = [];
attractor = zeros(3, 1);
x0_all = zeros(3, nTraj);
trajectories = trajectories(:,5:nPoints-5,:);
% When filter = true the next lines of code will apply a savitzky golay
% filter to your data (this is recommended for raw human demonstrations)
for i = 1:nTraj
    traj = trajectories(:,:,i);
    % if i==1
    %     continue
    % elseif i==3
    %     continue
    % elseif i==7
    %     continue
    % end
    
    % % Plot position/velocity Trajectories
    % att = [0; 0; 0];
    % vel_samples = 5; vel_size = 0.75; 
    % [h_data, h_att, ~] = plot_reference_trajectories_DS(trajectories(:,:,i), att, vel_samples, vel_size);
    if filter
        % Filter Trajectories and Compute Derivativess with Savitzky Golay filter
        %   traj: The trajectory you want to filter
        %   sample_step: subsample the traj before filtering
        %   nth_order :     max order of the derivatives 
        %   n_polynomial :  Order of polynomial fit
        %   window_size :   Window length for the filter
        traj = sgolay_filter_smoothing(trajectories(:,:,i), 5, 1, 4, 15);
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

est_options = [];
est_options.type             = 0;   % GMM Estimation Algorithm Type 

% If algo 1 selected:
est_options.maxK             = 20;  % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 30;  % Maximum Sampler Iterations
                                    % For type 0: 20-50 iter is sufficient
                                    % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics

% ====> PC-GMM computational complexity highly depends # data-points
% The lines below define the sub-sample variable you want to use
% on the trajectory datasets based on dataset size.
% You can play around with this to see the effect of dataset size on the 
% the different GMM estimation techniques
% Hint: 1->2 for 2D datasets, >2->3 for real 3D datasets
nb_data = length(Data);
sub_sample = 1;
draw_data = false;
% For LASA dataset
if (draw_data == false)
    if nb_data > 1500
        sub_sample = 8;    
    elseif nb_data > 1000
        sub_sample = 4;
    elseif nb_data > 500
        sub_sample = 2;
    end
     l_sensitivity = 2;
else % For Hand-drawn dataset
    if nb_data < 500
     sub_sample = 1;
    else 
     sub_sample = 2;
    end
    l_sensitivity = 5;
end

est_options.sub_sample       = sub_sample;


% Metric Hyper-parameters (for algo 0)
est_options.estimate_l       = 1;   % '0/1' Estimate the lengthscale, if set to 1
est_options.l_sensitivity    = l_sensitivity;   % lengthscale sensitivity [1-10->>100]
                                    % Default value is set to '2' as in the
                                    % paper, for very messy, close to
                                    % self-intersecting trajectories, we
                                    % recommend a higher value

est_options.length_scale     = [];  % if estimate_l=0 you can define your own
                                    % l, when setting l=0 only
                                    % directionality is taken into account

%%% These commands might need to be run in some machines                                    
% Give acces to Lasa developper if you are on mac
% ControlFlag = readlines('mac_setup/ControlFlag.txt');
% if ismac && (ControlFlag(1) == "LASADEVELOP = True") && (est_options.type ==0)
%     disp("Open Acces to LASA developper ")
%     system("sudo spctl --master-disable");
% end

% Fit GMM to Trajectory Data
[Priors, Mu, Sigma] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);

%% Generate GMM data structure for DS learning
clear ds_gmm; ds_gmm.Mu = Mu; ds_gmm.Sigma = Sigma; ds_gmm.Priors = Priors; 

% (Recommended!) Step 2.1: Dilate the Covariance matrices that are too thin
% This is recommended to get smoother streamlines/global dynamics
adjusts_C  = 1;
if adjusts_C  == 1
    if M == 2
        tot_dilation_factor = 1; rel_dilation_fact = 0.2;
    elseif M == 3
        tot_dilation_factor = 1; rel_dilation_fact = 0.75;
    end
    Sigma_ = adjust_Covariances(ds_gmm.Priors, ds_gmm.Sigma, tot_dilation_factor, rel_dilation_fact);
    ds_gmm.Sigma = Sigma_;
end

%  Visualize Gaussian Components and labels on clustered trajectories 
% Extract Cluster Labels
[~, est_labels] =  my_gmm_cluster(Xi_ref, ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, 'hard', []);

% Visualize Estimated Parameters
visualizeEstimatedGMM(Xi_ref, ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, est_labels, est_options);
title('GMM PDF contour ($\theta_{\gamma}=\{\pi_k,\mu^k,\Sigma^k\}$). Initial Estimate','Interpreter','LaTex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 3 (DS ESTIMATION): ESTIMATE SYSTEM DYNAMICS MATRICES  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DS OPTIMIZATION OPTIONS %%%%%%%%%%%%%%%%%%%%%% 
% Type of constraints/optimization 
constr_type = 2;      % 0:'convex':     A' + A < 0 (Same as SEDS, convex)
                      % 1:'non-convex': A'P + PA < 0 (Estimate P, nonconvex)
                      % 2:'non-convex': A'P + PA < Q (Pre-estimates P, Q <= -eps*I explicitly constrained)                                 
init_cvx    = 1;      % 0/1: initialize non-cvx problem with cvx solution, normally this is not needed
                      % but for some datasets with lots of points or highly non-linear it helps the 
                      % non-convex optimization converge faster. However, in some cases it might  
                      % bias the non-cvx problem too much and reduce
                      % reproduction accuracy.

if constr_type == 0 || constr_type == 1
    P_opt = eye(M);
else
    % P-matrix learning (Data shifted to the origin)
    % Assuming origin is the attractor (optimization works better generally)
    [Vxf] = learn_wsaqf(Data);
    P_opt = Vxf.P;
    fprintf('P matrix pre-estimated.\n');
end

%%%%%%%%  LPV system sum_{k=1}^{K}\gamma_k(xi)(A_kxi + b_k) %%%%%%%%  
if constr_type == 1
    [A_k, b_k, P_est] = optimize_lpv_ds_from_data(Data, zeros(M,1), constr_type, ds_gmm, P_opt, init_cvx);
    ds_lpv = @(x) lpv_ds(x-repmat(att, [1 size(x,2)]), ds_gmm, A_k, b_k);
else
    [A_k, b_k, ~] = optimize_lpv_ds_from_data(Data, att, constr_type, ds_gmm, P_opt, init_cvx);
    ds_lpv = @(x) lpv_ds(x, ds_gmm, A_k, b_k);
end

% Fill in plotting options
ds_plot_options = [];
ds_plot_options.sim_traj  = 1;            % To simulate trajectories from x0_all
ds_plot_options.x0_all    = x0_all;       % Intial Points
ds_plot_options.init_type = 'ellipsoid';  % For 3D DS, to initialize streamlines
                                          % 'ellipsoid' or 'cube'
ds_plot_options.nb_points = 30;           % No of streamlines to plot (3D)
ds_plot_options.plot_vol  = 0;            % Plot volume of initial points (3D)

[hd, hs, hr, x_sim] = visualizeEstimatedDS(Xi_ref, ds_lpv, ds_plot_options);
limits = axis;
switch constr_type
    case 0
        title('GMM-based LPV-DS with QLF', 'Interpreter', 'LaTex', 'FontSize', 20)
    case 1
        title('GMM-based LPV-DS with P-QLF (v0) ', 'Interpreter', 'LaTex', 'FontSize', 20)
    case 2
        title('GMM-based LPV-DS with P-QLF', 'Interpreter', 'LaTex', 'FontSize', 20)
end

if M == 2
    legend('Dataset trajectories', 'Learned trajectories')
elseif M == 3
    legend('Dataset trajectories', 'Learned DS')
end

% Compute RMSE on training data
rmse = mean(rmse_error(ds_lpv, Xi_ref, Xi_dot_ref));
fprintf('LPV-DS with (O%d), got velocity RMSE on training set: %d \n', constr_type+1, rmse);

% Compute e_dot on training data
edot = mean(edot_error(ds_lpv, Xi_ref, Xi_dot_ref));
fprintf('LPV-DS with (O%d), got velocity deviation (e_dot) on training set: %d \n', constr_type+1, edot);

% Compute DTWD between train trajectories and reproductions
if ds_plot_options.sim_traj
    nb_traj       = size(x_sim, 3);
    ref_traj_leng = size(Xi_ref, 2) / nb_traj;
    dtwd = zeros(1, nb_traj);
    for n=1:nb_traj
        start_id = round(1 + (n-1) * ref_traj_leng);
        end_id   = round(n * ref_traj_leng);
        dtwd(1,n) = dtw(x_sim(:,:,n)', Xi_ref(:,start_id:end_id)', 20);
    end
    fprintf('LPV-DS got DTWD of reproduced trajectories: %2.4f +/- %2.4f \n', mean(dtwd), std(dtwd));
end
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