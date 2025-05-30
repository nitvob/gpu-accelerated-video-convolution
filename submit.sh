#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_1g.5gb
#SBATCH -A cmsc416-class
#SBATCH -o cuda-video-%A.out
#SBATCH -J cuda-video
#SBATCH -t 00:01:00

# load necessary libraries
module load cuda opencv

# modules doesn't add lib64 path to LD_LIBRARY_PATH for some reason
# do it manually for cudart and opencv
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/cuda-11.6.2-eonihhhvlh4s2d6riyb7al2qivzn477u/lib64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/opencv-4.5.2-xxyodykxk3vuw64tlvm6sujgaxnctgep/lib64"

# run video-effect
./video-effect inputvideo.mp4 video-blur.mp4 blur 128 128
./video-effect inputvideo.mp4 video-edge.mp4 edge 128 128 100 actual-edge-100.csv
./video-effect inputvideo.mp4 video-sharpen.mp4 sharpen 128 128
./video-effect inputvideo.mp4 video-identity.mp4 identity 128 128