# CUDA-tutorial
Simple CUDA tutorial for Python

## Prerequisite
- CUDA-capable hardware (e.g., NVIDIA GPU)

## How to compile CUDA files
- A CUDA file (extension  ``*.cu``) needs to be compiled into a Parallel Thread Execution file (extension  ``*.cu``)
- All CUDA files are located in the folder ``CUDA-tutorial > pycuda_sourcecodes``
- Depending on the available hardware architecture, the script ``CUDA-tutorial > pycuda_sourcecodes > compile_pycuda_sourcecode.sh`` can be edited to utilize either Pascal (``sm_60``), Turing (``sm_75``), or Ampere (``sm_80`` or ``sm_86``)
- From ``CUDA-tutorial > pycuda_sourcecodes``, run the batch script ``compile_pycuda_sourcecode.sh``:
```sh
./compile_pycuda_sourcecode.sh name_of_the_cuda_file_to_compile.cu
```

## About CUDA programming
##### Core idea:
- Operations are performed in parallel by ``Threads``, as opposed to sequentially by ``for loops``
- ``Threads`` are organized within an upper-hierarchy of ``Blocks``, themselves organized within an upper-hierarchy of ``Grid``

##### Coordinates:
- The Grid is unique and therefore has no coordinates
- Blocks have 3D coordinates within the Grid ``(blockIDx.x, blockIDx.y, blockIDx.z)``
- Threads have 3D coordinates within a Block ``(threadIDx.x, threadIDx.y, threadIDx.z)``

##### Size:
- The Grid is defined by its 3D shape ``(gridDim.x, gridDim.y, gridDim.z)``
- Blocks are defined by their 3D shape ``(blockDim.x, blockDim.y, blockDim.z)``
- Thread are zero-dimensional and therefore have no shape
