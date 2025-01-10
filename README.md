# GPU-Accelerated Video Convolution

## Project Overview

I implemented GPU-accelerated video processing using convolution kernels to apply effects such as blurring, edge detection, sharpening, and identity transformations. My focus was to leverage the parallel computing capabilities of CUDA to efficiently process video frames. Each frame is convolved with an image kernel to achieve the desired effect, where each color channel (red, green, and blue) is processed independently. The implementation supports large-scale video processing and ensures efficient utilization of GPU resources.

This project involved:

- Designing and implementing a CUDA kernel (`convolveGPU`) for applying convolution operations on video frames.
- Developing a serial CPU-based implementation (`convolveCPU`) for correctness validation.
- Integrating OpenCV for handling video input and output.
- Conducting performance analysis to evaluate the impact of GPU optimizations, grid size, and block size.

## Technical Overview

The core of this project was optimizing the GPU kernel to map threads to image pixels. I used a striding mechanism to ensure full image coverage when there were more pixels than available threads. Each thread computes the convolution for multiple pixels, while border pixels are excluded to maintain correctness.

Video frames are processed in batches to optimize memory usage. By overlapping data transfer with computation using CUDA streams, GPU utilization is maximized.

Supported video effects include:

- **Blur**: Smoothens the image.
- **Edge Detection**: Highlights edges in the image.
- **Sharpen**: Enhances edges and details.
- **Identity**: Retains the original frame.

## Project Structure

The project is structured as follows:

- **`driver.cpp`**: The main driver program for reading video input, applying effects, and writing the output.
- **`video-effect.cu`**: Implements the CUDA kernel (`convolveGPU`) and associated GPU-based functions for applying convolution effects.
- **`Makefile`**: Contains build instructions for compiling the project.
- **`submit.sh`**: A script for submitting jobs to the Zaratan HPC cluster, including resource allocation and execution commands.
- **`analysis.pdf`**: Performance analysis report detailing execution times and performance trends for different block sizes.
- **`expected-edge-100.csv`**: Sample output file for validating the correctness of the edge detection effect.
- **`inputvideo.mp4`**: A sample input video file for testing the application.

## Prerequisites

To run this project, ensure the following are installed:

- **CUDA Toolkit 11.6 or later**
- **OpenCV 4.5 or later**
- A compatible GPU (e.g., NVIDIA A100)
- **GCC 9.4 or later**
- Access to a High-Performance Computing (HPC) cluster with GPU support (e.g., Zaratan at UMD).

## Installation

To set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/nitvob/gpu-accelerated-video-convolution.git
   cd gpu-accelerated-video-convolution
   ```
2. Load the required modules:
   ```bash
   module load cuda opencv
   ```
3. Set the necessary library paths:
   ```bash
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/cuda/lib64:/path/to/opencv/lib64"
   ```

## Setup Instructions

Compile the project using the provided Makefile:

1. Run the following command:

   ```bash
   make
   ```

   This will generate an executable named `video-effect`.

2. Verify the executable by running:
   ```bash
   ./video-effect --help
   ```

## How to Run

Run the program using the following syntax:

```bash
./video-effect <input_file> <output_file> <kernel_name> <grid_size_x> <grid_size_y> [optional:<frame_idx>] [optional:<frame_file>]
```

### Examples:

- To apply a blur effect:
  ```bash
  ./video-effect inputvideo.mp4 video-blur.mp4 blur 128 128
  ```
- To apply edge detection and dump frame 100 to a CSV:
  ```bash
  ./video-effect inputvideo.mp4 video-edge.mp4 edge 128 128 100 actual-edge-100.csv
  ```
- To apply sharpening:
  ```bash
  ./video-effect inputvideo.mp4 video-sharpen.mp4 sharpen 128 128
  ```
- To apply an identity transformation:
  ```bash
  ./video-effect inputvideo.mp4 video-identity.mp4 identity 128 128
  ```

## How to Test

Testing the project involves:

- **Verifying correctness**:
  Compare the output with expected results:
  ```bash
  diff actual-edge-100.csv expected-edge-100.csv
  ```
- **Performing a visual check**:
  Open the output MP4 files (e.g., `video-blur.mp4`, `video-edge.mp4`) in a media player to ensure the convolution was correctly applied.
- **Analyzing performance metrics**:
  Check the statistics printed to stdout:
  - Total time taken.
  - Frames processed per second.

## Performance Analysis

I analyzed the CUDA implementation to evaluate the impact of block sizes on execution time. Here are the results:

- **Optimal Block Size**: A block size of 16x16 achieved the best performance, significantly reducing execution time.
- **Performance Trends**:
  - Increasing the block size from 2x2 to 16x16 improved performance due to better GPU resource utilization.
  - Increasing the block size further to 32x32 caused performance degradation due to increased shared memory usage and occupancy limitations.

### Key Metrics

- **Batch Size**: Frames are processed in batches, optimizing memory usage while balancing throughput.
- **Data Transfer**: Overlapping data transfer with computation using CUDA streams enhances GPU utilization.
- **Execution Time**:
  - Block size 2x2: 1.2265 seconds.
  - Block size 16x16: 0.2143 seconds (optimal).
  - Block size 32x32: 0.3123 seconds.

## Skills and Knowledge Gained

This project was a valuable learning experience that strengthened my technical expertise in several areas. I gained hands-on experience designing and optimizing GPU kernels, implementing efficient memory usage, and utilizing CUDA streams for data transfer and computation. Working on video processing helped me understand how to apply convolution kernels for real-time effects and integrate OpenCV for video input and output.

Additionally, I developed skills in high-performance computing, including analyzing performance metrics, tuning grid and block sizes, and managing workloads on HPC clusters. Debugging and validation were key parts of this project, where I validated GPU results using serial CPU implementations and performed both visual and automated testing. These experiences improved my ability to work on performance-critical applications and deepened my understanding of GPU-based parallel programming.
