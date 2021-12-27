extern "C" { // ---> [kernel]

  __global__ void get_roi_cuda_sourcecode_row_major(
    float *roi,
    const float *dummy_data,
    const int *region_bot
    ){

    // Semantic parameters definition from the grid and blocks properties
    int dim_x = gridDim.x;          // Size of the "dummy_data" and "roi" arrays along the X dimension
    int dim_y = gridDim.y;          // Size of the "dummy_data" and "roi" arrays along the Y dimension
    int id_x = blockIdx.x;          // Coordinate along the X dimension
    int id_y = blockIdx.y;          // Coordinate along the Y dimension
    int id_thickness = threadIdx.x; // Relative coordinate along the depth dimension (i.e, in "roi", relative to "region_bot")

    // Indices (NOTE: Indexing is 1/ zero-based; 2/ row-major when data is taken from Python
    int pos_xy = id_x * dim_y + id_y;                                     // Position along the surface of "region_bot"
    int id_depth = id_thickness + region_bot[pos_xy];                     // Absolute depth coordinate (i.e, in "dummy_data")
    int pos_xyz_rel = id_thickness * dim_x * dim_y + id_x * dim_y + id_y; // Relative position (i.e., in "roi")
    int pos_xyz_abs = id_depth * dim_x * dim_y + id_x * dim_y + id_y;     // Absolute position (i.e, in "dummy_data")

    // Populate the output variable (Look mum, no for loop!!)
    roi[pos_xyz_rel] = dummy_data[pos_xyz_abs];

  } // ---> __global__ void get_roi_cuda_sourcecode(
} // ---> [kernel]
