#!/bin/sh
nvcc -ptx -arch 'sm_75' -O3 $1
