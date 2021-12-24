import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------------------------------------------------------
def nice_colorbar(im, axx):
  divider = make_axes_locatable(axx)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im, cax=cax)


# ----------------------------------------------------------------
def display_results(
  dummy_data, roi_gpu, roi_cpu, region_top, region_bot, t_cpu, t_gpu, slice_y, depth, length, dim_x, dim_y):

  fig, ax = plt.subplots(2, 2)

  axx = ax[0, 0]
  fig.sca(axx)
  im = axx.imshow(dummy_data[:, slice_y, :])
  plt.plot(region_top[slice_y, :], label='region top', color='darkorange')
  plt.plot(region_bot[slice_y, :], label='region bot', color='orchid')
  clim = im.properties()['clim']
  plt.title(
    'Data:\n' + r'(Depth $\times$ Dim X $\times$ Dim Y) = (' + \
    str(depth) + r'$\times$' + str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    'Shown here for slice Dim Y = ' + str(slice_y))
  plt.xlabel('Dim X')
  plt.ylabel('Depth')
  plt.legend()
  nice_colorbar(im, axx)

  axx = ax[1, 0]
  fig.sca(axx)
  im = axx.imshow(region_bot, cmap='inferno')
  plt.title(
    'Region bot:\n' + r'(Dim X $\times$ Dim Y) = (' + \
    str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    's.t. Region top = Region bot + ' + str(length))
  plt.xlabel('Dim X')
  plt.ylabel('Dim Y')
  nice_colorbar(im, axx)

  axx = ax[0, 1]
  fig.sca(axx)
  im = axx.imshow(roi_cpu[:, slice_y, :], vmin=clim[0], vmax=clim[1])
  plt.title(
    'ROI CPU [' + str(t_cpu) + 's]:\n' + r'(Length $\times$ Dim X $\times$ Dim Y) = (' + \
    str(length) + r'$\times$' + str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    'Shown here for slice Dim Y = ' + str(slice_y))
  plt.xlabel('Dim X')
  plt.ylabel('Length')
  nice_colorbar(im, axx)

  axx = ax[1, 1]
  fig.sca(axx)
  im = axx.imshow(roi_gpu[:, slice_y, :], vmin=clim[0], vmax=clim[1])
  plt.title(
    'ROI GPU [' + str(t_gpu) + 's]:\n' + r'(Length $\times$ Dim X $\times$ Dim Y) = (' + \
    str(length) + r'$\times$' + str(dim_x) + r'$\times$' + str(dim_y) + ')\n' + \
    'Shown here for slice Dim Y = ' + str(slice_y))
  plt.xlabel('Dim X')
  plt.ylabel('Length')
  nice_colorbar(im, axx)

  fig.set_dpi(300)
  fig.set_size_inches(20, 15, forward = True)
  fig.savefig('results_gpu_vs_cpu.png', bbox_inches = 'tight')
  plt.close('all')
