import sys
import numpy as np
from icecream import ic
import time
sys.path.insert(0, 'python_routines')
from python_routines.get_roi_cuda_handler import get_roi_cuda_handler
from python_routines.routines import generate_dummy_data
from python_routines.routines import generate_region_boundaries
from python_routines.routines import get_roi_cpu
from python_routines.routines import display_results


if __name__ == '__main__':

  # Parameters
  depth = 301         # Depth-wise dimension of the array "dummy_data"
  dim_x = 701         # X-wise dimension of the array "dummy_data"
  dim_y = 501         # Y-wise dimension of the array "dummy_data"
  half_thickness = 50 # Half-thickness of the sub-region that will be extracted along the depth
  half_jitter = 30    # Half amount of the possible depth-wise variation of the position of the extracted region
  slice_x = 21        # Slice of the 3D array that will be displayed for example

  # Derived parameters
  mid_depth = int(np.ceil(depth/2)) # Position mid-way along the depth axis
  thickness = 2*half_thickness +1      # Thickness of the region that will be extracted along depth

  # Generate a 3D dummy data, of size (depth, dim_x, dim_y)
  t = time.time()
  dummy_data = generate_dummy_data(depth, dim_x, dim_y)
  t_dummy = time.time() - t
  
  # Generate two 2D surfaces of size (dim_x, dim_y), s.t. "region_bot = region_bot + thickness"
  t = time.time()
  region_top, region_bot = generate_region_boundaries(dim_x, dim_y, mid_depth, half_jitter, half_thickness)
  t_regions = time.time() - t

  # Extract from "dummy_data" the values contained between "region_bot" and "region top" ...
  # ... and build the 3D array "roi_xxx" of size (thickness, dim_x, dim_y)

  # GPU (PyCUDA, loop-less implementation)
  t = time.time()
  roi_gpu = get_roi_cuda_handler(dummy_data, region_bot, thickness, dim_x, dim_y)
  t_gpu = time.time() - t

  # CPU (Python, three nested for-loops)
  t = time.time()
  roi_cpu = get_roi_cpu(dummy_data, region_bot, thickness, dim_x, dim_y)
  t_cpu = time.time() - t

  # Log
  ic(dummy_data.shape)
  ic(region_top.shape)
  ic(region_bot.shape)
  ic(roi_gpu.shape)
  ic(roi_cpu.shape)
  ic(t_dummy)
  ic(t_regions)
  ic(t_cpu)
  ic(t_gpu)
  ic(t_cpu / t_gpu)
  ic(np.array_equal(roi_gpu, roi_cpu))

  # Display
  display_results(
    dummy_data, roi_gpu, roi_cpu, region_top, region_bot, t_cpu, t_gpu, slice_x, depth, thickness, dim_x, dim_y)
