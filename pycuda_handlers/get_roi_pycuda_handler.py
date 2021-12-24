import pycuda.autoinit
import pycuda.driver as drv
import os
import numpy as np


def get_roi_pycuda_handler(dummy_data, region_bot, length, dim_x, dim_y, depth):

  # Define grids and blocks size
  grid_size = (dim_x, dim_y, 1) # 3D s.t. {1<=x<=65535, 1<=y<=65536, 1<=z<=65535}
  block_size = (length, 1, 1)   # 3D s.t. {1<=x<=1024; 1<=y<=1024; 1<=z<=64; x*y*x<=1024}

  # Initialize the array(s) that will be passed as input/output to the PyCUDA function
  roi = np.zeros((length, dim_x, dim_y), dtype = 'float32')

  # Get the PyCUDA module
  pycuda_module = drv.module_from_file(os.path.join('pycuda_sourcecodes', 'get_roi_pycuda_sourcecode.ptx'))

  # Get the PyCUDA kernel
  pycuda_kernel = pycuda_module.get_function('get_roi_pycuda_sourcode')

  # Call the kernel
  pycuda_kernel(
    drv.Out(roi),                          # <--- "drv.Out(.)": Input & Output (can be several)
    drv.In(dummy_data.astype(np.float32)), # <--- "drv.In(.)": Input only (can be several)
    drv.In(region_bot.astype(np.int32)),   # <--- Here we pass an array, and we specify its type
    drv.In(np.int32(depth)),               # <--- Here we pass a scalar, and we specify its type
    grid = grid_size,                      # <--- Grid size
    block = block_size)                    # <--- Block size

  return roi
