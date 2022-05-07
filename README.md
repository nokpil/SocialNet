# Social learning spontaneously emerges by searching optimal heuristics with deep reinforcement learning

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
python main.py --system $1 --spreader $2 --iter $3 --epochs $4 --n $5 --m $6 --Q $7 --constant $8 --noise $9 --indicator ${10:-""} 

```

The code automatically generates dataset if it isn't already exist at the 'data' folder. The trained model will be saved at 'result' folder.

### Argument descriptions
- system : Type of system. 'S1', 'S2', 'S3', 'P1', 'P2', 'P3'

