# SlimeMould
This is the repository to maintain the code for the following paper
"*A minimal model of cognition based on oscillatory and
reinforcement processes*". 


## Requirement
The code is written in Python, and the following packages are required. 

- numpy=1.24.3
- networkx=3.1
- matplotlib=3.7.2
- scipy=1.11.1

## Code
The following files are contained under the code folder. 

- main_model-cont.py: cycle graph generation, model with constant input/output, and result visualisation.
- main_model-ost.py: cycle graph generation, model with two oscillatory nodes (phase difference: $\pi$), and result visualisation.
- main_model-ost-phs.py: cycle graph generation, model with two oscillatory nodes (varying phase differences between $-\pi$ and $\pi$), and result visualisation (inc. bifurcation plots).
- main_hexagon.py: hexogon graph generation, model with multiple oscillatory nodes, and result visualisation (inc. videos).

## Citation
If you use the code in this repository for your publication, please cite us as follows. \
{\
*our bib here*\
}

## Contact
If you have any questions, please contact [Yu Tian](mailto:yu.tian@su.se).
