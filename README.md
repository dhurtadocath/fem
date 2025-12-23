# FEM - Computational Contact Mechanics and Machine Learning

This repository contains the code used for my PhD thesis, which focuses on two main topics:

1. **Minimization-Based Contact Mechanics**: Implementations of different minimization techniques (BFGS, L-BFGS, and Trust Region methods) to solve contact problems in quasi-static simulations.
2. **Neural Network-Based Contact Detection**: Multi-task neural network models for accelerating contact detection in rigid-body simulations.

## Repository Structure

```
FEM/
│── 1_Minimization_solvers/         # Code for minimization-based contact solvers
│── 2_Accelerated_Contact_Detection/ # Code for ANN-based contact detection
│── PyClasses/                       # Python classes used in the main scripts
│── Meshes/                           # Mesh data used for simulations
│── Other_work/                       # Additional or exploratory work
│── Paper1: Remedy_to_Newton_v0/      # Material related to a paper submission
│── .gitignore                        # Files to be ignored by git
│── environment.yml                    # Conda environment setup
│── fem-env.yml                        # Additional environment configuration
│── requirements.txt                    # Python dependencies
```

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/dhurtadocath/fem.git
cd fem
```

### 2. Set Up the Environment

We now provide multiple ways to set up the environment, optimized for different use cases:

#### Option A: Using Python Virtual Environment (Recommended)

##### Linux/macOS:
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt-get install build-essential libeigen3-dev python3.10-dev

# macOS:
brew install eigen

# Create and activate environment
./scripts/setup_env.sh [base|ml|viz|dev|all]
# Options:
#   base: Core dependencies only (default)
#   ml:   Base + Machine Learning dependencies
#   viz:  Base + Visualization dependencies
#   dev:  All dependencies + development tools
#   all:  All dependencies

# Activate the environment
source venv/bin/activate
# or
source activate_env.sh
```

##### Windows:
```powershell
# Ensure you have Visual Studio Build Tools and Eigen3 installed
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create and activate environment
.\scripts\setup_env_windows.ps1 [base|ml|viz|dev|all]

# Activate the environment
.\venv\Scripts\Activate.ps1
# or
.\activate_env.ps1
```

#### Option B: Using Docker (Best for Reproducibility)

```bash
# Build and run with Docker Compose
docker-compose up fem-base  # For base environment
docker-compose up fem-ml    # For ML environment with GPU support
docker-compose up fem-viz   # For visualization environment
docker-compose up fem-dev   # For development environment

# Or build manually
docker build -t fem-contact-mechanics:base --build-arg ENV_TYPE=base .
docker run -it -v $(pwd):/app fem-contact-mechanics:base
```

#### Option C: Using Legacy Conda (Not Recommended)
```bash
# Legacy method - slower and less reproducible
conda env create -f environment.yml
conda activate fem-env
pip install -r requirements.txt
```



### 3. Running the Code
Each of the main sections (`1_Minimization_solvers/` and `2_Accelerated_Contact_Detection/`) contains 2D and 3D examples that require specific input parameters.

#### **Minimization-Based Contact Mechanics**
- **2D Case:**
```bash
python pseudo2d.py --min_method BFGS --mesh 5 --plastic 0
```
  - Available meshes: `5` (5x5x5), `10`, `15`
  - Minimization methods: `BFGS`, `LBFGSnn` (nn is the number of iteration differences stored), `TR`, `TR-icho` (Trust regions with incomplete Cholesky decomposition as a preconditioner)

- **3D Case:**
```bash
python ContactPotato_Ex1.py --min_method BFGS --mesh 10 --plastic 0
```

#### **Neural Network-Based Contact Detection**
The scripts in `2_Accelerated_Contact_Detection/` follow the same execution principle but include an additional flag to determine if the Multi-Task ANN is used.

- **Example (3D with ANN disabled):**
```bash
python ContactPotato_Ex2.py --min_method BFGS --mesh 15 --plastic 0 --ann 0
```

## Compilation Notice
The material solver in `PyClasses/FEAssembly.py` uses compiled functions to compute the material contributions to the system. **Only the first time you run a script, the code will need to compile this, which may take several minutes.** After that, the compiled files remain in `PyClasses/` and will be reused in subsequent runs, significantly reducing execution time.

## Troubleshooting

### Common Issues

1. **fast_mfk module compilation fails**
   - This is expected on first run and can take 5-10 minutes
   - If it fails repeatedly, try running manually: `python PyClasses/compile_fast_mfk.py`
   - The module uses Numba JIT compilation which requires time for optimization

2. **Eigen3 not found**
   - Linux: `sudo apt-get install libeigen3-dev`
   - macOS: `brew install eigen`
   - Windows: Download from https://eigen.tuxfamily.org and add to include path

3. **C++ extension build fails**
   - Ensure you have a C++ compiler installed
   - Linux: `sudo apt-get install build-essential`
   - macOS: Install Xcode Command Line Tools
   - Windows: Install Visual Studio Build Tools

4. **Import errors after setup**
   - Make sure you activated the virtual environment
   - Try rebuilding: `pip install -e . --force-reinstall`

5. **GPU/CUDA issues with TensorFlow**
   - The ML environment assumes CPU by default
   - For GPU support, install appropriate CUDA/cuDNN versions

## Migration from Conda

The project has been migrated from Conda to standard Python virtual environments for:
- **Faster setup**: 2-5 minutes vs 10-30 minutes
- **Better reproducibility**: Exact version pinning
- **Easier deployment**: Standard pip packages
- **Smaller footprint**: ~500MB vs 2-3GB

If you need to use the old Conda environment, the `environment.yml` file is still available.

## Future Improvements
- Adding **unit tests** for core functionalities.
- Implementing **GitHub Actions** for automated testing.
- Improving documentation with detailed explanations of models and methods.
- Complete migration of all examples to the new environment structure.

## License
TBD

