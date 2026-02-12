# Vulkan Gaussian Splatting Viewer

A personal project to learn the **Vulkan API** and the fundamentals of **3D Gaussian Splatting**. This viewer was a practical exercise in moving from theory to a functional rendering pipeline.

![Vulkan](https://img.shields.io/badge/Vulkan-1.3-red) ![C++](https://img.shields.io/badge/C++-20-blue) ![macOS](https://img.shields.io/badge/Platform-macOS%20%2F%20MoltenVK-lightgrey)

![Splat Preview](https://github.com/Veltrynox/vulkan_triangle/raw/main/docs/splats_preview.gif)

---

The project runs on macOS via **MoltenVK**

### The Splatting Pipeline
* **PLY Parsing:** A custom loader to map raw data to GPU structures, handling quaternions for rotation and log-space scaling.
* **Math Kernels:** Logic to expand 3D ellipsoids into screen-aligned quads within the vertex shader.
* **SH Lighting:** Processing Spherical Harmonics coefficients to get view-dependent color.

Gaussian splatting requires back-to-front rendering for correct alpha accumulation. 
I implemented a CPU-based Depth Sorter that reorders splat indices relative to the camera. The resulting indices are uploaded to the GPU via Storage Buffers (SSBOs) to drive the draw calls.
