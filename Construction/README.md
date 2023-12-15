# Construction

Make sure to clone this repository to both your om2 directory and your scratch directory. The server has some strange kinks when trying to work out of scratch only, so it is recommended to keep all of the files that you edit and make changes to on your om2 directory, and have saved data and models be written to the scratch directory.

If you are not using openmind, you can ignore this section^^

## Install necessary packages

This code is built on Python 3.8.5. You'll definitely need Pytorch, Pygame, Scipy, Numpy, and Matplotlib. When you try to run the code it should inform you of any missing packages, and you can follow instructions online to install them


## Train an inference network

Optional: Delete `data/` in your scratch directory to regenerate training data.

You might have to create the folders `save/construction` within your scratch directory
```
mkdir save
mkdir save/construction
```

### Train L0 inference network
Run 
```
python train_construction_desire_pred.py --trainL0 --experiment-name <experiment-name> --num-data-train <num-data-train> --num-data-test <num-data-test> 
```

### Train L1 inference network
Run 
```
python train_construction_desire_pred.py --trainL1 --experiment-name <experiment-name> --num-data-train <num-data-train> --num-data-test <num-data-test> --l0-model-dir <path-to-L0-model>
```

For both training runs, you have the option of using BFS for planners or Manhattan Distance heuristic planners for the agents. By default, Manhattan Distance planners are enabled, but you can add the argument ```--useBFS``` to use BFS instead. I've tried to optimize this as much as I could, but there is still a bottleneck in the transition function of the construction environment. As such, I recommend only using this option when generating data for your large training runs, and using the heuristic planner for debugging. I would also appreciate help with debugging the transition function, deepcopying the state variables is the main bottleneck based on profilers.

There are additional arguments you can experiment with in the ```train_construction_desire_pred.py``` file.

The first time you train either nnL1 or nnL0 for a certain dataset length x training/testing pair, it will create a new dataset before it begins training.


## Test the inference network
Run
```
python nnL1_accuracy.py <beta-L0> <beta-L1>
```

For this to work, you'll once again need to replace lines within the file '''nnL1_accuracy.py''' with the path to your saved models directory and saved model paths. I've left my directories in the file as a guide, but you should substitute that with wherever you saved your models. This can be done between lines 123-154. Then, between lines 161-180, modify the arguments to load in an evaluation dataset as needed.

## State and Action Representation

### State Representation 
We represent the state of the construction environment by using multiple channels corresponding to different types of items. We end up with a 3-dimensional tensor to represent each state, with the dimensions 20 x 20 x 24, such that each cell in the 20 x 20 gridworld is represented as a one-hot vector. The vector representation of each cell indicates whether a cell has a wall, is empty space, is Alice without any blocks selected in her inventory, is Alice with one of 10 possible blocks in her inventory, is Bob without any blocks grabbed or is Bob with one of 10 possible blocks he can grab. 

### Action Representation
We represent the action an agent takes from a particular state as a one-hot vector associated with the following actions {Up, Down, Left, Right, Put-down, Stop}. An agent grabs a block automatically by moving onto the cell the block is located without having anything in its inventory, assuming another agent is not already at that coordinate. Thus, we do not need to explicitly encode this action and only need to give an agent the ability to drop items in its inventory.


## Collect data for human experiments

On your local machine (not openmind), run
```
python humanExperiment.py <participant_number> trial
```
This will create a separate window with the game. The human will play as the red robot representing the L1 agent. Their task is to infer which two blocks the blue agent (L0) wants to put together, and either help or hurt them based on the prompt indicated in the console. The controls for the human are described in the prompt at every timestep. Participant number should be an integer indicating which participant you are testing. Each participant generates 10 different datapoints (5 helping, 5 hurting), so failing to provide each participant with a unique number will result in previous participants having their data overwritten. The "trial" argument is optional. If it is there, the user will be able to test the dynamics of the environment once without their data being collected. If it isn't, all data will be recorded.