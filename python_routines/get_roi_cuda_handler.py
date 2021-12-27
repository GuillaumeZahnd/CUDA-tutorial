import pycuda.autoinit
import pycuda.driver as drv
import os
import numpy as np


def get_roi_cuda_handler(dummy_data, region_bot, thickness, dim_x, dim_y):

  # Define grid and block size
  grid_size = (dim_x, dim_y, 1)  # 3D s.t. {1<=x<=65535, 1<=y<=65536, 1<=z<=65535}
  block_size = (thickness, 1, 1) # 3D s.t. {1<=x<=1024; 1<=y<=1024; 1<=z<=64; x*y*x<=1024}

  # Initialize the array(s) that will be passed as input/output to the PyCUDA function
  roi = np.zeros((thickness, dim_x, dim_y), dtype = 'float32')

  # Get the PyCUDA module
  cuda_module = drv.module_from_file(os.path.join('cuda_sourcecodes', 'get_roi_cuda_sourcecode_row_major.ptx'))

  # Get the PyCUDA kernel
  cuda_kernel = cuda_module.get_function('get_roi_cuda_sourcecode_row_major')

  # Call the kernel
  cuda_kernel(
    drv.Out(roi),                          # <--- "drv.Out(.)": Input & Output (can be several)
    drv.In(dummy_data.astype(np.float32)), # <--- "drv.In(.)": Input only (can be several)
    drv.In(region_bot.astype(np.int32)),   # <--- Here we pass an array, and we specify its type
    grid = grid_size,                      # <--- Grid size
    block = block_size)                    # <--- Block size

  return roi
