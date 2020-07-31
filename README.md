# pi-IW

Implementation of pi-IW and AlphaZero in distributed tensorflow. It uses OpenAI Gym's environments.

To run pi-IW on Pong (single process):

    python3 run.py path exp_name --env-id PongWrapped-v0 --planner PolicyGuidedIW --lookahead LookaheadReturns --features NNLastHiddenBool --algorithm SupervisedPolicy

For AlphaZero:

    python3 run.py path exp_name --env-id PongWrapped-v0 --planner MCTSAlphaZero --lookahead LookaheadCounts --algorithm SupervisedPolicyValue

See run.py path experiment_name -h for all other options.

Atari games that end with 'Wrapped-v0' stack 4 consecutive grayscale images into a 4-channel image. Also, a frameskip can be specified for these games.

All experiments of the paper were done using the single process version above.

For distributed training, we need to run many instances of launch_task.py. We provide a helper script to automatically run many processes in the same machine. Learning is distributed in many workers that compute stale gradients from their copy of the network parameters. They can perform a gradient descent step updating the global (shared) parameters in a synchronous or asyncrhonous manner.


##### Update (31/07/2020)
We corrected a bug that altered the input of the neural network for atari games. This affects the results of Table 2 of the paper.