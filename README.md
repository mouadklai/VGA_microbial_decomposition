# Voxel-Based Approach for Simulating Microbial Decomposition of Organic Matter in Soil

This project provides an implementation of a voxel-based approach for simulating microbial decomposition of organic matter in soil using 3D computed tomography (CT) images.

## Overview

This approach models microbial decomposition within a soil structure represented by 3D voxel grids derived from CT images. The simulation incorporates biological and physical processes such as microbial growth, substrate consumption, taking into account diffusion of organic matter.

### Key Features

- **Voxel-Based Representation**: The soil structure is discretized into a grid of voxels based on 3D CT images.
- **Microbial Dynamics**: Simulates microbial activity using numerical models informed by experimental data.
- **Diffusion Processes**: Includes the diffusion of organic compounds across the soil matrix using a simple Fick law (less computationnally epensive).
- **Machine Learning Integration**: Uses machine learning to approximate diffusional conductance coefficients for pore network geometrical models for an enhanced accuracy. 
## Get the code

To get started, clone the repository:

```bash
git [clone https://github.com/mouadklai/VGA_microbial_decomposition.git
