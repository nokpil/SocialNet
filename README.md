# ConservNet
This repository contains the official PyTorch implementation of the RL framework from:

**Social learning spontaneously emerges by searching optimal heuristics with deep reinforcement learning** (https://arxiv.org/abs/2204.12371)
by Seungwoong Ha and Hawoong Jeong.

**Abstract** : How have individuals of social animals in nature evolved to learn from each other, and what would be the optimal strategy for such learning in a specific environment? Here, we address both problems by employing a deep reinforcement learning model to optimize the social learning strategies (SLSs) of agents in a cooperative game in a multi-dimensional landscape. Throughout the training for maximizing the overall payoff, we find that the agent spontaneously learns various concepts of social learning, such as copying, focusing on frequent and well-performing neighbors, self-comparison, and the importance of balancing between individual and social learning, without any explicit guidance or prior knowledge about the system. The SLS from a fully trained agent outperforms all of the traditional, baseline SLSs in terms of mean payoff. We demonstrate the superior performance of the reinforcement learning agent in various environments, including temporally changing environments and real social networks, which also verifies the adaptability of our framework to different social settings. 

## Requirements
- Python 3.6+
- Pytorch 1.0+ (written for 1.8)

## Run experiments
To replicate the experiments by running
```
python ConservNet.py --system $1 --spreader $2 --iter $3 --epochs $4 --n $5 --m $6 --Q $7 --constant $8 --noise $9 --indicator ${10:-""} 

```
Similarly, we have implemented Siamese Neural Network (SNN) from  [S.  J.  Wetzel,  R.  G.  Melko,  J.  Scott,  M.  Panju,    and
6V. Ganesh, Physical Review Research2, 033499 (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033499).

```
python Siam.py --system $1 --iter $3 --epochs $4 --n $5 --m $6 --noise $7 --indicator ${8:-""} 

```

The code automatically generates dataset if it isn't already exist at the 'data' folder. The trained model will be saved at 'result' folder.

### Argument descriptions
- system : Type of system. 'S1', 'S2', 'S3', 'P1', 'P2', 'P3'
- iter : iteration number. Perform same experiment multiple times (default: 1)
- spreader : Type of spreader. 'L1', 'L2', 'L8' (default : L2)
- epochs : total number of training epochs (default: 10000) 
- n : batch number (default: 20)
- m : batch size (default: 200)
- Q : spreading constant (default: 1.0)
- R : max norm of injected noise (default: 1.0)
- noise : scale of added noise (default : 0.0)
- indicator : string which will be concatenated to the name of the saved model. (default: '')

## System specifications
- S1, S2, S3 are the simulated system with invariants written on the paper.
- P1 : Simulated Lotka-Volterra equations with a prey and a pradator.
- P2 : Simulated Kepler problem with two bodies.
- P3 : Real double pendulum data from [M. Schmidt and H. Lipson, science 324, 81 (2009)](https://science.sciencemag.org/content/324/5923/81).
  - If you set the system as P3, batch_number and batch_size will be fixed to 1 and 818, ignoring arguments n and m.
  - Note that P3 for SNN is not available.
