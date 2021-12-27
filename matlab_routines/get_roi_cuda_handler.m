% NOTE: 
% When a GPU command is called for the first time by a given MATLAB session, the process may take several minutes
% This phenomenon is likely to be related to the notion of "just-in-time compilation", and is case-specific
% After the first call to a GPU command, all subsequent calls from the same MATLAB sessions are free from this specific delay


function roi = get_roi_cuda_handler(dummy_data, region_bot, depth, thickness, dim_x, dim_y)

  % Define the CUDA module and kernel
  cuda_module_path_and_file_name = fullfile('cuda_sourcecodes', 'get_roi_cuda_sourcecode_column_major.ptx');
  cuda_kernel_name = 'get_roi_cuda_sourcecode_column_major';
  
  % Get the CUDA kernel from the CUDA module
  cuda_kernel = parallel.gpu.CUDAKernel(cuda_module_path_and_file_name, 'float *, const float *, const int *, const int *, const int *', cuda_kernel_name);

  % Define the grid and block size
  cuda_kernel.GridSize = [dim_x, dim_y, 1];        % 3D s.t. {1<=x<=65535, 1<=y<=65536, 1<=z<=65535}
  cuda_kernel.ThreadBlockSize = [thickness, 1, 1]; % 3D s.t. {1<=x<=1024; 1<=y<=1024; 1<=z<=64; x*y*x<=1024}

  % Initialize GPU array(s) that will be passed as input/output to the CUDA function
  roi = zeros([thickness, dim_x, dim_y], 'single');
 
  % Call the kernel (the input variables must respect their expected type)
  roi = feval(cuda_kernel, roi, single(dummy_data), int32(region_bot), depth, thickness);
  
  % Gather the output back from GPU to CPU
  roi = gather(roi);

end
