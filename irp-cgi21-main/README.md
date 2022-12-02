# Independent Research Project - Solving Partial Differential Equations (PDEs) using Artificial Intelligence (AI) Libraries
#### Chukwume Ijeh
#### Supervisors: Christopher C. Pain, Claire E. Heaney and Chen Boyang

## Overview
With the exponential growth of the use of AI and AI technology today, there has been the recent development of "AI" computers capable of delivering the wall-clock compute performance of many tens to hundreds of GPUs. This has spurred the interest of researchers, especially in the field of Computational Fluid Dynamics (CFD), to exploit this new possibility. 

Currently, CFD codes are currently being written using low-level languages, such as FORTRAN/C++. However, if the possibility of writing CFD codes using an AI library, such as PyTorch, and running such codes on the "AI" computer, the following issues currently being faced in CFD modeling can be addressed:
1. Currently, only expert CFD developers can write CFD code. This is attributed to how complex CFD codes are, and how they have to be rewritten to suit different hardware architectures. This limits accessibility and parallel scalability between programmers.
2. The computational cost and time of running CFD codes is enormous. 

Through the use of AI libraries, we aim to simplify various steps involved in writing CFD codes, using convolutional neural networks (CNN). This will help increase accessibility and parallel scalability between programmers. Also, running these newly developed codes on "AI" computers will help address and reduce the computational costs of CFD codes. 

In a broader sense, simpler and faster CFD codes will help the simpler development of digital twins in various industries. The digital twins can then be used to optimise systems, form error measures, assimilate data and quantify uncertainty in a relatively straight-forward manner using this approach.


## Info
This repository contains all the code associated with this research, with the following files:
- GNN_FEM_Structured.ipynb: Containing graph neural network (GNN) code for structured data
- GNN_FEM_Unstructured.ipynb: Containing GNN code for unstructured data
- Requirements.txt: Containing all the requirements for this research
- Structured_Data_Multigrid.ipynb: Containing the geometric multigrid solver for structured data
- Unstructured_Data_Multigrid.ipynb: Containing the geometric multigrid solver for unstructured data
- shape_functions.f90: Module to carry out a finite element (FE) discretization on a 2D mesh
- space_filling_decomp_new.f90: Module to generate space filling curves (SFCs) orderings

And the following folders:
- Unstructured_Meshes: Containing the meshes used in this research
- GNN_Structured_Results: Containing results on a structured 128x128 mesh using a GNN
- GNN_Unstructured_Results: Containing results on an unstructured 64x64 mesh using a GNN

To compile the `shape_functions.f90` and `space_filling_decomp_new.f90` modules, run the following command in terminal:
- python3 -m numpy.f2py -c shape_functions.f90 -m shape_functions
- python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_curves































































