import sys
import numpy as np
from icecream import ic
import time
sys.path.insert(0, 'pycuda_handlers')
from pycuda_handlers.get_roi_pycuda_handler import get_roi_pycuda_handler
from display_results import display_results


# ----------------------------------------------------------------
def generate_dummy_data(depth, dim_x, dim_y):

  # A 3D array "dummy_data" is generated, of size (depth, dim_x, dim_y)
  # The array has anisotropic dimensions to be more interesting, because it better enforce correct indexing
  # The array is populated with values that correspond to the Euclidean distance from the corner (0, 0, 0)
  # The array is corrupted by some noise to justify the "float32" type

  pos_x = np.linspace(0, depth, depth)
  pos_y = np.linspace(0, dim_x, dim_x)
  pos_z = np.linspace(0, dim_y, dim_y)
  grid_x, grid_y, grid_z = np.meshgrid(pos_x, pos_y, pos_z, indexing='ij')
  dummy_data = np.sqrt(np.power(grid_x, 2) + np.power(grid_y, 2) + np.power(grid_z, 2)).astype('float32')
  np.random.seed(1)
  dummy_data += np.random.normal(loc=0.0, scale=1.0, size=(depth, dim_x, dim_y))
 
  return dummy_data


# ----------------------------------------------------------------
def generate_region_boundaries(dim_x, dim_y, mid_x, half_jitter, half_length):

  # A pair of 2D surfaces "region_top" and "region_bot" are generated, of size (dim_x, dim_y)
  # These two surfaces are separated by a constant distance "length = 2*half_length +1"

  np.random.seed(1)
  region_mid = mid_x + np.random.randint(low=-half_jitter, high=half_jitter+1, size=(dim_x, dim_y), dtype=int)
  region_top = region_mid + half_length
  region_bot = region_mid - half_length

  return region_top, region_bot


# ----------------------------------------------------------------
def get_roi_cpu(dummy_data, region_bot, region_top, length, dim_x, dim_y):

  # From "dummy_data", the 3D array "roi_cpu" is constructed, of size (length, dim_x, dim_y)
  # The array is defined as the volume encompassed by "region_bot" and "region_top"
  # Note that this algorithm implementation is unnecessarily sub-optimal on purpose

  roi_cpu = np.zeros((length, dim_x, dim_y)).astype('float32')
  for x in range(dim_x):
    for y in range(dim_y):
      for l in range(length):
        roi_cpu[l, x, y] = dummy_data[region_bot[x, y] + l, x, y]

  return roi_cpu


# ----------------------------------------------------------------
if __name__ == '__main__':

  # Parameters
  depth = 301      # Depth-wise dimension of the array "dummy_data"
  dim_x = 701      # X-wise dimension of the array "dummy_data"
  dim_y = 501      # Y-wise dimension of the array "dummy_data"
  half_length = 50 # Half-length of the sub-region that will be extracted along depth
  half_jitter = 30 # Half amount of the possible variation of the position of the extracted region along depth
  slice_y = 21     # Slice of the 3D array that will be displayed for example

  # Derived parameters
  mid_depth = int(np.ceil(depth/2)) # Position mid-way along the depth axis
  length = 2*half_length +1         # Length of the region that will be extracted along depth

  # Generate a 3D dummy data, of size (depth, dim_x, dim_y)
  t = time.time()
  dummy_data = generate_dummy_data(depth, dim_x, dim_y)
  t_dummy = time.time() - t
  
  # Generate two 2D surfaces of size (dim_x, dim_y), s.t. "region_bot = region_bot + length"
  t = time.time()
  region_top, region_bot = generate_region_boundaries(dim_x, dim_y, mid_depth, half_jitter, half_length)
  t_regions = time.time() - t

  # Extract from "dummy_data" the values contained between "region_bot" and "region top" ...
  # ... and build the 3D array "roi_xxx" of size (length, dim_x, dim_y)

  # GPU (PyCUDA, loop-less implementation)
  t = time.time()
  roi_gpu = get_roi_pycuda_handler(dummy_data, region_bot, length, dim_x, dim_y, depth)
  t_gpu = time.time() - t

  # CPU (Python, three nested for-loops)
  t = time.time()
  roi_cpu = get_roi_cpu(dummy_data, region_bot, region_top, length, dim_x, dim_y)
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
    dummy_data, roi_gpu, roi_cpu, region_top, region_bot, t_cpu, t_gpu, slice_y, depth, length, dim_x, dim_y)
