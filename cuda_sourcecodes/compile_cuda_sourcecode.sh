#!/bin/sh
nvcc -ptx -arch $1 -O3 $2
