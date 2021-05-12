disp('======= KITTI 2015 Benchmark Demo =======');
clear all; close all; 
# dbstop error;

% error threshold
tau = [3 0.05];

% stereo demo
disp('Load and show disparity map ... ');
D_est = disp_read('data/disp_est.png');
D_gt  = disp_read('data/disp_gt.png');
d_err = disp_error(D_gt,D_est,tau);

disp(d_err)
