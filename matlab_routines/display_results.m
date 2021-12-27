function display_results(dummy_data, roi_gpu, roi_cpu, region_top, region_bot, t_gpu, t_cpu, slice_y, depth, thickness, dim_x, dim_y)

  figure(1); clf;
  
  ax1 = subplot(2, 2, 1); imagesc(squeeze(dummy_data(:, slice_y, :))); axis image; colorbar; colormap(ax1, 'parula');
  hold on;
    p1 = plot(region_top(slice_y, :), 'color', [0.850, 0.320, 0.098]);
    p2 = plot(region_bot(slice_y, :), 'color', [0.750, 0.000, 0.750]);
  hold off;
 legend([p1, p2], 'Region top', 'Region bot');
  title({...
    'Data:',...
    ['(Depth $\times$ Dim X $\times$ Dim Y) = (' num2str(depth) '$\times$' num2str(dim_x) '$\times$' num2str(dim_y) ')'],...
    ['Shown here for slice Dim Y = ' num2str(slice_y)]}, 'interpreter', 'latex');
  xlabel('Dim X');
  ylabel('Depth');
  
  ax2 = subplot(2, 2, 3); imagesc(region_bot); axis image; colorbar; colormap(ax2, 'cool');
  title({...
    'Region bot:',...
    ['(Dim X $\times$ Dim Y) = (' num2str(dim_x) '$\times$' num2str(dim_y) ')'],...
    ['s.t. Region top = Region bot + ' num2str(thickness)]}, 'interpreter', 'latex');  
  xlabel('Dim X');
  ylabel('Dim Y');

  subplot(2, 2, 2); imagesc(squeeze(roi_gpu(:, slice_y, :))); axis image; colorbar; colormap(ax1, 'parula');
  title({...
    ['ROI GPU [' num2str(t_gpu) 's]:'],...
    ['(Thickness $\times$ Dim X $\times$ Dim Y) = (' num2str(thickness) '$\times$' num2str(dim_x) '$\times$' num2str(dim_y) ')'],...
    ['Shown here for slice Dim Y = ' num2str(slice_y)]}, 'interpreter', 'latex');
  xlabel('Dim X');
  ylabel('Thickness');

  subplot(2, 2, 4); imagesc(squeeze(roi_cpu(:, slice_y, :))); axis image; colorbar; colormap(ax1, 'parula');
  title({...
    ['ROI CPU [' num2str(t_cpu) 's]:'],...
    ['(Thickness $\times$ Dim X $\times$ Dim Y) = (' num2str(thickness) '$\times$' num2str(dim_x) '$\times$' num2str(dim_y) ')'],...
    ['Shown here for slice Dim Y = ' num2str(slice_y)]}, 'interpreter', 'latex');
  xlabel('Dim X');
  ylabel('Thickness');
  
  print('results_MATLAB_gpu_vs_cpu.png', '-dpng');
  
end
