#!/bin/sh

for i in *.cu
do
  echo "nvcc -ptx -arch $1 -O3 $i"
  nvcc -ptx -arch $1 -O3 $i
done
