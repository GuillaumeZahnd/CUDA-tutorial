import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
def generate_region_boundaries(dim_x, dim_y, mid_x, half_jitter, half_thickness):

  # A pair of 2D surfaces "region_top" and "region_bot" are generated, of size (dim_x, dim_y)
  # These two surfaces are separated by a constant distance "thickness = 2*half_thickness +1"

  np.random.seed(1)
  region_mid = mid_x + np.random.randint(low=-half_jitter, high=half_jitter+1, size=(dim_x, dim_y), dtype=int)
  region_top = region_mid + half_thickness
  region_bot = region_mid - half_thickness

  return region_top, region_bot


# ----------------------------------------------------------------
def get_roi_cpu(dummy_data, region_bot, thickness, dim_x, dim_y):

  # From "dummy_data", the 3D array "roi_cpu" is constructed, of size (thickness, dim_x, dim_y)
  # The array is defined as the volume encompassed by "region_bot" and "region_top"
  # Note that this algorithm implementation is unnecessarily sub-optimal on purpose

  roi_cpu = np.zeros((thickness, dim_x, dim_y)).astype('float32')
  for x in range(dim_x):
    for y in range(dim_y):
      for l in range(thickness):
        roi_cpu[l, x, y] = dummy_data[region_bot[x, y] + l, x, y]

  return roi_cpu


# ----------------------------------------------------------------
def display_results(
  dummy_data, roi_gpu, roi_cpu, region_top, region_bot, t_cpu, t_gpu, slice_x, depth, thickness, dim_x, dim_y):

  fig, ax = plt.subplots(2, 2)

  axx = ax[0, 0]
  fig.sca(axx)
  im = axx.imshow(dummy_data[:, slice_x, :])
  plt.plot(region_top[slice_x, :], label='region top', color='darkorange')
  plt.plot(region_bot[slice_x, :], label='region bot', color='orchid')
  clim = im.properties()['clim']
  plt.title(
    'Data:\n' + r'(Depth $\times$ Dim X $\times$ Dim Y) = (' + \
    str(depth) + r'$\times$' + str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    'Shown here for slice Dim X = ' + str(slice_x))
  plt.xlabel('Dim Y')
  plt.ylabel('Depth')
  plt.legend()
  nice_colorbar(im, axx)

  axx = ax[1, 0]
  fig.sca(axx)
  im = axx.imshow(region_bot, cmap='inferno')
  plt.title(
    'Region bot:\n' + r'(Dim X $\times$ Dim Y) = (' + \
    str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    's.t. Region top = Region bot + ' + str(thickness))
  plt.xlabel('Dim Y')
  plt.ylabel('Dim X')
  nice_colorbar(im, axx)

  axx = ax[0, 1]
  fig.sca(axx)
  im = axx.imshow(roi_gpu[:, slice_x, :], vmin=clim[0], vmax=clim[1])
  plt.title(
    'ROI GPU [' + str(t_gpu) + 's]:\n' + r'(Thickness $\times$ Dim X $\times$ Dim Y) = (' + \
    str(thickness) + r'$\times$' + str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    'Shown here for slice Dim X = ' + str(slice_x))
  plt.xlabel('Dim Y')
  plt.ylabel('Thickness')
  nice_colorbar(im, axx)

  axx = ax[1, 1]
  fig.sca(axx)
  im = axx.imshow(roi_cpu[:, slice_x, :], vmin=clim[0], vmax=clim[1])
  plt.title(
    'ROI CPU [' + str(t_cpu) + 's]:\n' + r'(Thickness $\times$ Dim X $\times$ Dim Y) = (' + \
    str(thickness) + r'$\times$' + str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    'Shown here for slice Dim X = ' + str(slice_x))
  plt.xlabel('Dim Y')
  plt.ylabel('Thickness')
  nice_colorbar(im, axx)

  fig.set_dpi(300)
  fig.set_size_inches(20, 15, forward = True)
  fig.savefig('results_Python_gpu_vs_cpu.png', bbox_inches = 'tight')
  plt.close('all')


# ----------------------------------------------------------------
def nice_colorbar(im, axx):
  divider = make_axes_locatable(axx)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im, cax=cax)
