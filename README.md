Voxel-Based Approach for Simulating Microbial Decomposition in Soil
This repository contains the implementation of the voxel-based approach presented in our paper:

📄 A voxel-based approach for simulating microbial decomposition in soil: Comparison with LBM and improvement of morphological models

Overview
This project provides a computational framework to simulate microbial decomposition of organic matter in soil using 3D computed tomography (CT) images. Unlike traditional geometric models, this approach directly uses voxel information from CT images to construct the simulation space, offering improved accuracy without requiring parameter calibration.

The model incorporates biological and physical processes, including:

Microbial growth and substrate consumption

Diffusion of organic compounds

Transformation of the soil system into a two-phase model (liquid–solid)

🔑 Key Features
Voxel-Based Representation: Soil microstructure is discretized into a 3D voxel grid directly derived from CT images.

Reaction–Diffusion Modeling: Microbial activity is simulated using a validated system of ordinary differential equations coupled with diffusion equations based on Fick’s law.

Efficient Diffusion Simulation: Compared to Lattice Boltzmann Method (LBM), our approach reduces computational cost by up to 75% while maintaining comparable accuracy.

Machine Learning Integration: Implements a stochastic gradient descent approach to approximate diffusional conductance coefficients in pore network geometrical models (PNGM), improving their precision.

Two-Phase System Handling: After separating the liquid, air, and solid phases, the method simplifies the system to liquid–solid interactions for computational efficiency.

📥 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/mouadklai/VGA_microbial_decomposition.git
cd VGA_microbial_decomposition
🚀 Usage
Coming soon – detailed instructions on preparing input CT images, running simulations, and reproducing the results from the paper will be added here.

📚 Citation
If you use this code in your research, please cite:

Klai M, Monga O, Jouini MS, Pot V.
A voxel-based approach for simulating microbial decomposition in soil: Comparison with LBM and improvement of morphological models.
PLOS ONE. 2025 Mar 3;20(3):e0313853.
https://doi.org/10.1371/journal.pone.0313853

🤝 Contributing
Contributions and suggestions are welcome! Please open an issue or submit a pull request if you’d like to help improve this project.

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.
