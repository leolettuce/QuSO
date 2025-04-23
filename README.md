# QuSO - Quantum Simulation-Based Optimization of a Cooling System 
This repository contains the implementation of the Quantum Simulation-Based Optimization (QuSO) algorithm for finding the optimal cooling system.
Details about the particular problem and the implementation can be found in the paper: https://arxiv.org/abs/2504.15460

## More Information
We have implemented QuSO in PennyLane and Classiq. This repository only contains the PennyLane code.
Check out the Classiq code here: http://short.classiq.io/cooling_systems_optimization

If you are interested in the differences or the particular strengths of both quantum software frameworks, check out the notebook in the "comparison" folder.

## Folder Structure
- comparison: contains a brief comparison between PennyLane and Classiq.
- experiments: contains the numerical experiments presented in the paper.
- OpenModelica: contains the scripts to validate our linear system with OpenModelica simulations.
- full_algorithm: contains the implementation of the entire QuSO algorithm.
- tutorial_notebooks: contains the jupyter notebooks used in the tutorial at QCE24 TUT24 (https://qce.quantum.ieee.org/2024/program/tutorials-abstracts/)
  
## Requirements
- numpy
- scipy
- matplotlib
- pennylane
- pennylane-lightning
- pyqsp
