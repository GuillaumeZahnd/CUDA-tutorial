clearvars;
close all;
clc;
set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('matlab_routines');


% Parameters
depth = 301;         % Depth-wise dimension of the array "dummy_data"
dim_x = 701;         % X-wise dimension of the array "dummy_data"
dim_y = 501;         % Y-wise dimension of the array "dummy_data"
half_thickness = 50; % Half-thickness of the sub-region that will be extracted along the depth
half_jitter = 30;    % Half amount of the possible depth-wise variation of the position of the extracted region
slice_x = 21;        % Slice of the 3D array that will be displayed for example

% Derived parameters
mid_depth = ceil(depth/2); % Position mid-way along the depth axis
thickness = 2*half_thickness +1; % Thickness of the region that will be extracted along depth

% Generate a 3D dummy data, of size (depth, dim_x, dim_y)
tic;
dummy_data = generate_dummy_data(depth, dim_x, dim_y);
t_dummy = toc;

% Generate two 2D surfaces of size (dim_x, dim_y), s.t. "region_bot = region_bot + thickness"
tic;
[region_top, region_bot] = generate_region_boundaries(dim_x, dim_y, mid_depth, half_jitter, half_thickness);
t_regions = toc;

% Extract from "dummy_data" the values contained between "region_bot" and "region top" ...
% ... and build the 3D array "roi_xxx" of size (thickness, dim_x, dim_y)
  
% GPU (CUDA, loop-less implementation)
tic;
roi_gpu = get_roi_cuda_handler(dummy_data, region_bot, depth, thickness, dim_x, dim_y);
t_gpu = toc;

% CPU (MATLAB, three nested for-loops)
tic;
roi_cpu = get_roi_cpu(dummy_data, region_bot, thickness, dim_x, dim_y);
t_cpu = toc;

% Log
disp(['dummy_data: ' num2str(size(dummy_data))]);
disp(['region_top: ' num2str(size(region_top))]);
disp(['region_bot: ' num2str(size(region_bot))]);
disp(['roi_gpu: ' num2str(size(roi_gpu))]);
disp(['roi_cpu: ' num2str(size(roi_cpu))]);
disp(['t_dummy: ' num2str(t_dummy) 's']);
disp(['t_regions: ' num2str(t_regions) 's']);
disp(['t_cpu: ' num2str(t_cpu) 's']);
disp(['t_gpu: ' num2str(t_gpu) 's']);
disp(['t_cpu / t_gpu: ' num2str(t_cpu / t_gpu)]);
disp(['isequal(roi_gpu, roi_cpu): ' num2str(isequal(roi_gpu, roi_cpu))]);

% Display
display_results(...
  dummy_data, roi_gpu, roi_cpu, region_top, region_bot, t_gpu, t_cpu, slice_x, depth, thickness, dim_x, dim_y);
