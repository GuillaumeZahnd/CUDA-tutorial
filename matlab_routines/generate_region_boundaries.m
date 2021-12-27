function [region_top, region_bot] = generate_region_boundaries(dim_x, dim_y, mid_depth, half_jitter, half_thickness)

  % A pair of 2D surfaces "region_top" and "region_bot" are generated, of size (dim_x, dim_y)
  % These two surfaces are separated by a constant distance "thickness = 2*half_thickness +1"

  rng(1);
  region_mid = mid_depth + randi([-half_jitter, +half_jitter], [dim_x, dim_y]); 
  region_top = region_mid + half_thickness;
  region_bot = region_mid - half_thickness;

end
