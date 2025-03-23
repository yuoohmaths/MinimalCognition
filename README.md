# Minimal Model of Cognition
This is the repository to maintain the code for the following paper
"Gyllingberg Linnéa†, Tian Yu† and Sumpter David J. T., [*A minimal model of cognition based on oscillatory and current-based reinforcement processes*](http://doi.org/10.1098/rsif.2024.0402), J. R. Soc. Interface, 22 (2025), rsif.2024.0402.". 


## Requirement
The code is written in Python, and the following packages are required. 

* [NumPy](https://github.com/numpy/numpy) = 1.24.3 (Matrix Manipulation)
* [NetworkX](https://github.com/networkx/networkx) = 3.1 (Based Graph library)
* [Matplotlib](https://github.com/matplotlib/matplotlib) = 3.7.2 (Visulisation)
* [SciPy](https://github.com/scipy/scipy) = 1.11.1 (ODE Solver)

## Code
The following files are contained under the code folder. 

- main_model-cont.py: cycle graph generation, the model with constant input/output, and result visualisation.
- main_model-ost.py: cycle graph generation, the model with two oscillatory nodes (phase difference: $\pi$), and result visualisation.
- main_model-ost-phs.py: cycle graph generation, the model with two oscillatory nodes (varying phase differences between $-\pi$ and $\pi$), and result visualisation (inc. bifurcation plots).
- main_model-ost-phAs.py: cycle graph generation, the model with two oscillatory nodes (varying phase differences between $-\pi$ and $\pi$ and amplitude ratios between $1/10$ and $10$), and result visualisation (inc. table of resulting graphs).
- main_model-ost-phths.py: cycle graph generation, the model with two oscillatory nodes (varying phase differences between $-\pi$ and $\pi$ and frequency ratio between $1/10$ and $10$), and result visualisation (inc. table of resulting graphs).
- main_hexagon.py: hexagon graph generation, the model with multiple oscillatory nodes, and result visualisation (inc. videos).

The following folders are contained under the code folder.

- data: large graph (constructed by hexagon tiling and random noise). 
- results: videos generated from main_hexagon.py with $3$ and $5$ oscillatory nodes.

## Citation
If you use this code in your research, please considering cite our paper:

```
@article{gts2024minimalcognition,
  title={A minimal model of cognition based on oscillatory and reinforcement processes},
  author={Gyllingberg*, L. and Tian*, Y. and Sumpter, D.},
  journal = {Journal of The Royal Society Interface},
  volume = {22},
  number = {222},
  pages = {rsif.2024.0402},
  year = {2025},
  doi = {10.1098/rsif.2024.0402}
}
```
## Contact
If you have any questions, please contact [Yu Tian](mailto:yu.tian.research@gmail.com).
