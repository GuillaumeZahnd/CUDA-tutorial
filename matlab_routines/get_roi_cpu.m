function roi_cpu = get_roi_cpu(dummy_data, region_bot, thickness, dim_x, dim_y)

  % From "dummy_data", the 3D array "roi_cpu" is constructed, of size (thickness, dim_x, dim_y)
  % The array is defined as the volume encompassed by "region_bot" and "region_top"
  % Note that this algorithm implementation is unnecessarily sub-optimal on purpose

  roi_cpu = zeros(thickness, dim_x, dim_y);
  for x = 1:dim_x
    for y = 1:dim_y
      for l = 1:thickness
        roi_cpu(l, x, y) = dummy_data(region_bot(x, y) + l, x, y);
      end
    end
  end
  roi_cpu = single(roi_cpu);
end
