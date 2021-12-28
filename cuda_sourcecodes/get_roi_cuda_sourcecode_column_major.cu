extern "C" { // ---> [kernel]

  __global__ void get_roi_cuda_sourcecode_column_major(
    float *roi,
    const float *dummy_data,
    const int *region_bot,
    const int *depth_pt
    ){

    // Obtain the scalar value for pointers input linked to zero-D arrays
    int depth = depth_pt[0];

    // Semantic parameters definition from the grid and blocks properties
    int dim_x = gridDim.x;          // Size of the "dummy_data" and "roi" arrays along the X dimension
    int id_x = blockIdx.x;          // Coordinate along the X dimension
    int id_y = blockIdx.y;          // Coordinate along the Y dimension
    int thickness = blockDim.x;     // Size of the "roi" array along the depth dimension
    int id_thickness = threadIdx.x; // Relative coordinate along the depth dimension (i.e, in "roi", relative to "region_bot")

    // Indices (NOTE: Indexing is 1/ zero-based; 2/ column-major when data is taken from MATLAB
    int pos_xy = id_y * dim_x + id_x;                                             // Position along the surface of "region_bot"
    int id_depth = id_thickness + region_bot[pos_xy];                             // Absolute depth coordinate (i.e, in "dummy_data")
    int pos_xyz_rel = id_y * dim_x * thickness + id_x * thickness + id_thickness; // Relative position (i.e., in "roi")
    int pos_xyz_abs = id_y * dim_x * depth + id_x * depth + id_depth;             // Absolute position (i.e, in "dummy_data")

    // Populate the output variable (Look mum, no for loop!!)
    roi[pos_xyz_rel] = dummy_data[pos_xyz_abs];

  } // ---> __global__ void get_roi_cuda_sourcecode(
} // ---> [kernel]
