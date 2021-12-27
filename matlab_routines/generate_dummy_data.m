function dummy_data = generate_dummy_data(depth, dim_x, dim_y)

  % A 3D array "dummy_data" is generated, of size (depth, dim_x, dim_y)
  % The array has anisotropic dimensions to be more interesting, because it better enforce correct indexing
  % The array is populated with values that correspond to the Euclidean distance from the corner (1, 1, 1)
  % The array is corrupted by some noise to justify the "float32" type

  pos_depth = linspace(1, depth, depth);
  pos_x = linspace(1, dim_x, dim_x);
  pos_y = linspace(1, dim_y, dim_y);
  [grid_depth, grid_x, grid_y] = ndgrid(pos_depth, pos_x, pos_y);
  dummy_data = sqrt(grid_depth.^2 + grid_x.^2 + grid_y.^2);
  rng(1);
  dummy_data = dummy_data + randn(depth, dim_x, dim_y);

end