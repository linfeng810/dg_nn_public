# AI4PDEs -- Implementing discontinuous Galerkin method with graph neural networks

This repository is part of a research project called AI4PDEs. Specifically, here it contains the discontinuous Galerkin Finite Element Method (DG-FEM) on unstructured mesh implemented with graph neural networks and convolutional neural networks. For the structured mesh solver, please refer to https://github.com/bc1chen/AI4PDE.

## How to use
### Prerequisites
-	Python 3.10 in a conda environment
-	Python packages listed in requirements_pip.txt
-	A Fortran compiler supported by NumPy f2py; see https://numpy.org/doc/stable/f2py/
### First-time set-up
The source codes are in several directories:
-	```z00-fortran-src```: Fortran codes for generating a space-filling curve, finding global connectivity, and finding coarse grid operators with a space-filling curve.
-	```z02-implicit/z06-DGtoCG```: A solver for the steady state diffusion problems. Only the sub-directory z06-DGtoCG is valid. The rest are deprecated.
-	```z05-ns```: A fluid-structure interaction solver. The fluid equation is incompressible Navier-Stokes equation; the structure equation is for large-deformation hyper-elastic materials.
-	```z01, z03,``` and ```z04``` are deprecated.

When running the code for the first time, the Fortran code needs to be compiled. Please follow the instructions:
-	Get into the solver directory (```z02*/z06*``` or ```z05*```).
-	Activate the conda environment that satisfies the prerequisites.
-	Type ```compile_f90.sh``` in the command line to compile the Fortran source in ```z00-fortran-src```.

### Run a simulation
For the diffusion solver, get into directory ```z02*/z06*/```
-	Generate the mesh with ```Gmsh```. The mesh should use 3rd order elements; Dirichlet boundaries should be in the first physical group; Neumann boundaries should be in the second physical group.
-	Boundary conditions and right-hand-side source term for the diffusion equation are set up in ```bc_f.py```.
-	Run the code with ```python3 main.py```

For the fluid-structure interaction solver, it is possible to solve a pure fluid, a pure structure, or a fluid-structure interaction problem. Get into directory ```z05-ns/```
-	The input mesh should mark the physical group in Gmsh as ''fluid'', or ''solid'', or mark two physical groups as ''fluid'' and ''solid'' for an FSI problem. When an FSI problem is modelled, the ''fluid'' physical group index should be smaller than the ''solid'' physical group index.
> **_NOTE:_** The mesh should also define the boundary conditions in the following order: fluid Dirichlet boundaries, fluid Neumann boundaries, solid Dirichlet boundaries, and solid Neumann boundaries. Otherwise, the code won’t recognise the correct boundaries.
-	Boundary conditions and right-hand-side source term for the diffusion equation are set up in ```bc_f.py```.
-	Problem settings: initial condition, material properties, and solver (linear/non-linear) settings are in ```config.py```.
-	Run the code with ```python3 main.py```

Note that running a simulation for the first time will take longer due to internal processing of the input mesh. The processed results will be stored as numpy arrays in ```*.npy``` files and reused in the future run. If a fresh run is wanted, the ```*.npy``` files should be removed from the directory where the mesh is stored.

## Examples
### Diffusion
-	A 2D analytical solution
-	A 3D analytical solution
-	Diffusion in a nozzle geometry
### Incompressible fluid
-	(to be published later)
### Hyperelastic
-	(to be published later)
### Fluid-structure interaction
-	(to be published later)

## Reference
Papers in preparation / under review
1.	Diffusion: Implementing the discontinuous-Galerkin finite element method using graph neural networks (under review, preprint link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4698813)
2.	Fluid dynamics: (in preparation)
3.	Structure statics: (in preparation)
4.	Fluid-structure interaction: (in preparation)
## Contact
Linfeng Li (l.li20 at imperial dot ac dot uk)

