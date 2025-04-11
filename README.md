# QuSO - Quantum Simulation-Based Optimization of a Cooling System 
This repository contains the implementation of the Quantum Simulation-Based Optimization (QuSO) algorithm for finding the optimal cooling system.
Details about the particular problem and the implementation can be found in the paper: [Insert arXiv link]

## More Information
We have implemented QuSO in PennyLane and Classiq. This repository only contains the PennyLane code.
Check out the Classiq code here: http://short.classiq.io/cooling_systems_optimization

If you are interested in the differences or the particular strengths of both quantum software frameworks, check out the notebook in the "comparison" folder.

## Folder Structure
- experiments: contains the numerical experiments presented in the paper.
- comparison: contains a brief comparison between PennyLane and Classiq.
- full_algorithm: contains the implementation of the entire QuSO algorithm.
  
## Requirements
- numpy
- scipy
- matplotlib
- pennylane
- pennylane-lightning
- pyqsp
