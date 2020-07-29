# pi-IW
Implementation of the paper [Deep Policies for Width-Based Planning in Pixel Domains](https://arxiv.org/abs/1904.07091), appearing in the Proceedings of the 29th International Conference on Automated Planning and Scheduling (ICAPS 2019).

pi-IW is an on-line planning algorithm that enhances Rollout IW by incorporating an action selection policy, resulting in an *informed* width-based search. It interleaves a planning and a learning step:
* Planning: expands a tree using Rollout IW guided by a policy estimate.
* Learning: a target policy is extracted from the planned trajectories and is used to update the policy estimate (supervised learning). 


## On-line planning
To illustrate how on-line planning works, and the benefit of adding a guiding policy, we provide examples for planning in MDPs in a simple corridor task:
* [One planning step](planning_step.py) of Rollout IW (off-line)
* [On-line Rollout IW](online_planning.py): interleaving planning steps with action executions, without any policy guidance or learning
* [On-line planning and learning](online_planning_learning.py): pi-IW using BASIC and dynamic features

## Experiments
We compare our algorithm with AlphaZero, and demonstrate that pi-IW has superior performance in simple, sparse-reward environments. For instance, to run pi-IW with dynamic features in the two walls environment:
```
python3 piIW_alphazero.py --algorithm pi-IW-dynamic --seed 1234 --env GE_MazeKeyDoor-v2
```  
See the help (-h) section for more details.

For atari games, use the deterministic version of the gym environments, which can be specified by selecting v4 environments (e.g. "Breakout-v4").

Important: this repository is a reimplementation of the algorithm in Tensorflow eager (2.0 compatible), which is more clear, intuitive, and easier to modify and debug. The results of the paper were obtained from a previous implementation in tensorflow graph (v1.4), which can be found in a separate [branch](https://github.com/aig-upf/pi-IW/tree/original). This previous version of the code allows to parallelize the algorithm using one parameter server and many workers, using distributed tensorflow. 

## Installation
* Install the requirements (numpy, tensorflow and gym packages)
* Make sure that [gridenvs](https://github.com/aig-upf/gridenvs) is added to the python path.



##### Update (31/07/2020)
We corrected a bug that altered the input of the neural network for atari games. This affects the results of Table 2 of the paper.