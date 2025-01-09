# GPU-Accelerated Video Convolution

## Project Overview

This project implements GPU-accelerated video processing using convolution kernels to apply effects such as blurring, edge detection, sharpening, and identity transformations. The main focus is to leverage the parallel computing capabilities of CUDA to efficiently process video frames. Each frame is convolved with an image kernel to achieve the desired effect, where each color channel (red, green, and blue) is processed independently. The implementation supports large-scale video processing and ensures efficient utilization of GPU resources.

The project includes:

- A CUDA kernel (`convolveGPU`) for applying convolution operations on video frames.
- A serial CPU-based implementation (`convolveCPU`) for correctness validation.
- Integration with OpenCV for video input/output handling.
- Performance analysis to evaluate the impact of GPU optimizations, grid size, and block size.

## Technical Overview

- **GPU Kernel Implementation**: The CUDA kernel maps threads to image pixels using a striding mechanism to handle more pixels than available threads. Each thread computes the convolution for multiple pixels, ensuring full image coverage. The implementation avoids border pixels to maintain correctness.
- **Batch Processing**: Video frames are processed in batches to optimize memory usage. CUDA streams are used for overlapping data transfer and computation, maximizing GPU utilization.
- **Supported Effects**:
  - **Blur**: Smoothens the image.
  - **Edge Detection**: Highlights edges in the image.
  - **Sharpen**: Enhances edges and details.
  - **Identity**: Retains the original frame.

## Project Structure

- **`driver.cpp`**: Contains the main driver program for reading video input, applying effects, and writing the output.
- **`video-effect.cu`**: Implements the CUDA kernel (`convolveGPU`) and associated GPU-based functions for applying convolution effects.
- **`Makefile`**: Provides the build instructions for compiling the project.
- **`submit.sh`**: A script to submit the job to the Zaratan HPC cluster, including resource allocation and execution commands.
- **`analysis.pdf`**: Performance analysis report detailing the execution times and performance trends for different block sizes.
- **`expected-edge-100.csv`**: Sample output file for validating the correctness of the edge detection effect.
- **`inputvideo.mp4`**: Sample input video file for testing the application.

## Prerequisites

Ensure the following software and libraries are installed:

- **CUDA Toolkit 11.6 or later**
- **OpenCV 4.5 or later**
- A compatible GPU (e.g., NVIDIA A100)
- **GCC 9.4 or later**
- Access to a High-Performance Computing (HPC) cluster with GPU support (e.g., Zaratan at UMD).

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Ensure the required modules are available on your system:
   ```bash
   module load cuda opencv
   ```
3. Set the necessary library paths:
   ```bash
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/cuda/lib64:/path/to/opencv/lib64"
   ```

## Setup Instructions

1. Compile the project using the provided Makefile:

   ```bash
   make
   ```

   This will generate an executable named `video-effect`.

2. Verify the executable:
   ```bash
   ./video-effect --help
   ```

## How to Run

Run the program using the following syntax:

```bash
./video-effect <input_file> <output_file> <kernel_name> <grid_size_x> <grid_size_y> [optional:<frame_idx>] [optional:<frame_file>]
```

### Examples:

1. Apply a blur effect:

   ```bash
   ./video-effect inputvideo.mp4 video-blur.mp4 blur 128 128
   ```

2. Apply edge detection and dump frame 100 to a CSV:

   ```bash
   ./video-effect inputvideo.mp4 video-edge.mp4 edge 128 128 100 actual-edge-100.csv
   ```

3. Apply sharpening:

   ```bash
   ./video-effect inputvideo.mp4 video-sharpen.mp4 sharpen 128 128
   ```

4. Apply an identity transformation:
   ```bash
   ./video-effect inputvideo.mp4 video-identity.mp4 identity 128 128
   ```

## How to Test

1. Verify correctness by comparing the output with expected results:
   ```bash
   diff actual-edge-100.csv expected-edge-100.csv
   ```
2. Open the output MP4 files (e.g., `video-blur.mp4`, `video-edge.mp4`) in a media player to visually verify that the convolution was correctly applied.
3. Check performance statistics printed to stdout:
   - Total time taken.
   - Frames processed per second.

## Performance Analysis

The CUDA implementation was analyzed to evaluate the impact of block sizes on execution time. The results showed:

- **Optimal Block Size**: A block size of 16x16 achieved the best performance, reducing execution time significantly.
- **Performance Trends**:
  - As the block size increased from 2x2 to 16x16, execution time decreased due to better GPU resource utilization.
  - Further increasing the block size to 32x32 led to performance degradation due to increased shared memory usage and occupancy limitations.

### Key Performance Metrics

- **Batch Size**: Frames are processed in batches, optimizing memory usage while balancing throughput.
- **Data Transfer**: Overlapping data transfer with computation using CUDA streams enhances GPU utilization.
- **Execution Time**:
  - Block size 2x2: 1.2265 seconds.
  - Block size 16x16: 0.2143 seconds (optimal).
  - Block size 32x32: 0.3123 seconds.

The implementation demonstrates significant speedup over a serial CPU implementation while highlighting the importance of tuning GPU configurations for optimal performance.
