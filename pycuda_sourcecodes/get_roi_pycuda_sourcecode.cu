extern "C" { // ---> [kernel]

  __global__ void get_roi_pycuda_sourcode(
    float *roi,
    const float *dummy_data,
    const int *region_bot
    ){

    // Semantic parameters definition from the grid and blocks properties
    int dim_x = gridDim.x;      // Size of the "dummy_data" and "roi" arrays along the X dimension
    int dim_y = gridDim.y;      // Size of the "dummy_data" and "roi" arrays along the Y dimension
    int id_x = blockIdx.x;      // Coordinate along the X dimension
    int id_y = blockIdx.y;      // Coordinate along the Y dimension
    int id_z_rel = threadIdx.x; // Relative coordinate along the Z dimension (i.e, in "roi", relative to "region_bot")

    // Indices (NOTE: 1/ indexing is zero-based; 2/ indexing is row-major from Python and column-major from MATLAB)
    int pos_xy = id_x * dim_y + id_y;                                 // Position along the surface of "region_bot"
    int id_z_abs = id_z_rel + region_bot[pos_xy];                     // Absolute Z coordinate (i.e, in "dummy_data")
    int pos_xyz_rel = id_z_rel * dim_x * dim_y + id_x * dim_y + id_y; // Relative position (i.e., in "roi")
    int pos_xyz_abs = id_z_abs * dim_x * dim_y + id_x * dim_y + id_y; // Absolute position (i.e, in "dummy_data")

    // Populate the output variable (Look mum, no for loop!!)
    roi[pos_xyz_rel] = dummy_data[pos_xyz_abs];

  } // ---> __global__ void get_roi_pycuda_sourcode(
} // ---> [kernel]
