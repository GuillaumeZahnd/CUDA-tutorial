# README

## About this repository

##### Purpose:
This repository offers a simple tutorial to concretely understand the basic principles of ``CUDA programming``. Example scripts are provided for both ``Python and MATLAB`` languages.
##### Toy example:
Let us consider a 3D volume of size ``(depth, dim_x, dim_y)``. The objective of this dummy operation is to extract a smaller 3D volume of size ``(thickness, dim_x, dim_y), s.t. thickness < depth``. This toy example is solved first via a GPU-accelerated parallel implementation, then via a CPU-based nested for loops implementation.

## Prerequisite
- See: [https://docs.nvidia.com/cuda/index.html](https://docs.nvidia.com/cuda/index.html)
- CUDA-capable hardware (e.g., NVIDIA GPU)
- Python and/or MATLAB
- CUDA-capable software environment (please check your own compilers, libraries, and toolkits)
- Note for struggling Debian/Fedora/Arch/... Linux users: as of today, Ubuntu is considered as being one of the most CUDA-compatible distro

## How to compile CUDA files
- See: [https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation)
- All CUDA files are located in the folder ``CUDA-tutorial > cuda_sourcecodes``
- A CUDA file (``.cu``) must be compiled via ``NVCC`` (Nvidia CUDA Compiler) into a Parallel Thread Execution file ( ``.ptx``)
- Identify the codename ``sm_xy`` that matches your GPU architecture (e.g., Turing is ``sm_75``):
- Run this command to compile the file ``name_of_the_cuda_file.cu`` into ``name_of_the_cuda_file.ptx`` (replace ``sm_xy`` by the value adapted to your GPU architecture):
```sh
./compile_pycuda_sourcecode.sh name_of_the_cuda_file.cu sm_xy
```
- Run this command to compile all CUDA files contained in ``CUDA-tutorial > cuda_sourcecodes`` (replace ``sm_xy`` by your the value adapted to your GPU architecture):
```sh
./batch_compile_pycuda_sourcecode.sh sm_xy
```

## How to run the example scripts
- For Python, run:
```sh
CUDA-tutorial > main_python.py
```
- For MATLAB, run:
```sh
CUDA-tutorial > main_matlab.m
```

## About CUDA acceleration
##### Core idea:
- Operations are carried out by the ``GPU``, as opposed to the ``CPU``
- Operations are performed in parallel by ``Threads``, as opposed to sequentially by ``nested for loops``
- ``Threads`` are organized within an upper-hierarchy of ``Blocks``, themselves organized within an upper-hierarchy of ``Grid``

##### Coordinates design:
- The ``Grid`` is unique and therefore has no coordinates
- ``Blocks`` have 3D coordinates within the Grid ``(blockIDx.x, blockIDx.y, blockIDx.z)``
- ``Threads`` have 3D coordinates within a Block ``(threadIDx.x, threadIDx.y, threadIDx.z)``

##### Size design:
- The ``Grid`` is defined by its 3D shape ``(gridDim.x, gridDim.y, gridDim.z)``
- ``Blocks`` are defined by their 3D shape ``(blockDim.x, blockDim.y, blockDim.z)``
- ``Threads`` are zero-dimensional and therefore have no shape

##### More advanced implementation techniques (not needed in this toy example and not covered in this tutorial):
- Use ``atomicAdd(int* address, int val)`` to prevent race conditions between different threads within the same thread block
- Use ``__syncthreads()`` to enforce a block-level synchronization barrier of parallel threads

##### Performance optimization  (disregarded in this tutorial):
- The number of threads per block should be a round multiple of the warp size, which is 32 on all current hardware
- When possible, the total number of threads should be augmented to keep the total number of blocks low

## Useful GPU-related Linux-based command lines
```sh
$ lspci | grep VGA
01:00.0 VGA compatible controller: NVIDIA Corporation TU106 [GeForce RTX 2060 SUPER] (rev a1)
```

```sh
$ watch -1 nvidia-smi
Every 1,0s: nvidia-smi
Mon Dec 27 12:29:10 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P5    12W / 175W |    148MiB /  7982MiB |     39%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1286      G   /usr/lib/xorg/Xorg                147MiB |
+-----------------------------------------------------------------------------+
```

```sh
$ sudo lshw -C display
*-display                 
   description: VGA compatible controller
   product: TU106 [GeForce RTX 2060 SUPER]
   vendor: NVIDIA Corporation
   physical id: 0
   bus info: pci@0000:01:00.0
   version: a1
   width: 64 bits
   clock: 33MHz
   capabilities: pm msi pciexpress vga_controller bus_master cap_list rom
   configuration: driver=nvidia latency=0
   resources: irq:138 memory:a4000000-a4ffffff memory:90000000-9fffffff memory:a0000000-a1ffffff ioport:3000(size=128) memory:a5000000-a507ffff
```

```sh
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```
