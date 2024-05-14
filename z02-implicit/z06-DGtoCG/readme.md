# AI4PDE solver for diffusion problems

In this directory are the main codes for an AI4PDE solver on the unstructured mesh for the diffusion problems. The solver is implemented with PyTorch. It contains convolutional and graph neural networks. No training is required.

The PDE being solved here is,

$$ \nabla^2 u = f$$

The numerical method is discontinuous Galerkin method with symmetric interior penalty scheme. The linear solver can be choosen from
- Multigrid stand-alone solver
- Multigrid preconditioned conjugate gradient solver
- Multigrid preconditioned GMRES solver with restart

## Example 1 -- A 2D analytical solution

The domain is a unit square $[0,1]^2$. All boundaries are of Dirichlet type.
The top boundary has $\sin(\pi x)$ condition; the rest boundaries are 0.

The problem has an analytical solution: $u = \sin(\pi x) \sinh(\pi y) / \sinh(\pi)$.

To run the problem, first get into ```z02*/z06*/z20-square-mesh```, then run bash ```generate_mesh.sh``` to generate a mesh. 

Then go to ```z02*/z06*```, modify the problem settings in ```config.py```:
- ```filename = '20-square-mesh/square_high_order.msh'```
- ```problem = 'diff-test'```

Run the code with ```python3 main.py```. (Don't forget to activate your conda environment)



## Example 2 -- A 3D analytical solution

The domain is a unit cube $[0,1]^3$. All boundaries are of Dirichlet type.
A manufactured solution $u=\exp(-x-y-z)$ is sought here. A right-hand side source term $f=-3\exp(-x-y-z)$ is inserted to the Poisson's equation to get this solution. Boundary conditions are computed from the manufactured solution.

To run the problem, first get into ```z02*/z06*/z21-cube-mesh```, then run bash ```generate_mesh.sh``` to generate a mesh. 

Then go to ```z02*/z06*```, modify the problem settings in ```config.py```:
- ```filename = 'z21-cube-mesh/cube_ho_poi.msh'```
- ```problem = 'diff-test'```

Run the code with ```python3 main.py```. (Don't forget to activate your conda environment)


## Example 3 -- Diffusion process in a nozzle

This is a stationary diffusion process in a nozzle shaped domain. For details of the domain and problem settings. To run the problem, ```Gmsh > 4.11``` is required to generate the mesh: ```gmsh -3 nozzle.geo```.

Then go to ```z02*/z06*```, modify the problem settings in ```config.py```:
- ```filename = $(path to the mesh you just generated)```
- ```problem = 'nozzle'```

Run the code with ```python3 main.py```. (Don't forget to activate your conda environment)

## Visulisation of the results

Results files will be stored in the same directory where the mesh file is. Results files
include:
- ```*.vtu``` file that can be opened with paraview
- ```*.pt``` file that stores the field values in a torch tensor file, this can be loaded into the code as initial condition.


## Use refined mesh
If one wishes to use a refined mesh for this problem, they can 
- open the ```*.geo``` with Gmsh, 
- generate a coarse mesh by clicking ```1D```, ```2D```, (and ```3D``` if for 3D problems.)
- click ```Refine by splitting```, 
- ```Set order 3```
- save the mesh.

You are also welcome to modify the mesh size directly in the ```*.geo``` file.
