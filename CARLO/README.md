# CARLO Driving Simulator

If you are not using openmind, you can ignore the following paragraph:

Make sure to clone this repository to both your om2 directory and your scratch directory. The server has some weird kinks when trying to work out of scratch only, so it is recommended to keep all of the files that you edit and make changes to on your om2 directory and have saved data and models be written to the scratch directory.

## Install necessary packages

This code is built on Python 3.8.5. I do not have a complete list of packages, but you'll definitely need Pytorch, Pygame, Scipy, Numpy, and Matplotlib. When you try to run the code it should inform you of any missing packages, and you can follow the instructions online to install them


## Train an inference network

Optional: Delete ```data/``` in your scratch directory to regenerate training data.

You might have to create the folders ```save/construction``` within your scratch directory
```
mkdir save
mkdir save/construction
```

**Sweep training**

If you want to run a sweep of experiments, run
```
bash run_car_sweep.sh
```
Within the ```car_sweep.py``` file you should see a number of arguments being iterated over. Each one will be submitted as a separate job on openmind, and will train either the state recognition model, the nnL0 goal recognition model or the nnL1 goal recognition model through their corresponding training files. You will have to modify the file within the ```car_utils/slurm.py``` file for specifics on how to submit jobs on your own cluster.


### Train State Recognition network
Run 
```
python train_belef_nn.py --experiment-name <experiment-name> --num-data-train <num-data-train> --num-data-test <num-data-test> 
```

### Train L0 Goal Recognition network

First, go to the file ```train_car_action_pred.py``` and update **line 109** with the correct path to the directory your state recognition models are saved.

Run 
```
python train_car_action_pred.py --experiment-name <experiment-name> --num-data-train <num-data-train> --num-data-test <num-data-test> 
```

### Train L1 Goal Recognition network

First, go to the file ```train_car_nnL1.py``` and update **line 126** with the correct path to the directory your state recognition models are saved. Then, update **line 123** with the correct path to the directory your L0 goal recognition model is saved.

Run 
```
python train_car_nnL1.py --experiment-name <experiment-name> --num-data-train <num-data-train> --num-data-test <num-data-test> 
```

### Train L1 Action Prediction network

First, go to the file ```train_car_nnL1.py``` and update **line 126** with the correct path to the directory your state recognition models are saved. Then, update **line 123** with the correct path to the directory your L0 goal recognition model is saved.

Run 
```
python train_car_nnL1.py --predictActions --experiment-name <experiment-name> --num-data-train <num-data-train> --num-data-test <num-data-test> 
```

There are additional arguments you can experiment with in each of the files described above. For instance, in training the L1 action prediction network you have the option to train using cross entropy losses, as we did in the paper as a baseline.

The first time you train any of the networks for a certain dataset length x training/testing pair, it will create a new dataset before it begins training (you have the option to specify the save directory).

## Test Inference Models

To evaluate the performance of your trained L1 models, you first need to make modifications to the file ```test_scenario2.py``` 

1. Between lines 361-408, substitute the appropriate variables with the path to the directory that contains your trained models (i.e. state recognition, goal recognition, etc.)
2. Between lines 411-426, customize the arguments for the data generation and evaluation process (i.e. # of cars, beta, planner depth, etc.)

Then, run
```
python test_scenario2.py
```

You can also plot different accuracy metrics by first modifying the ```accuracyViewer.py``` file.

1. Between lines 111-143, substitute the appropriate variables with the path to the directory that contains your trained models (i.e. state recognition, goal recognition, etc.)
2. Between lines 145-185, modify the arguments to the dataset used to evaluate and plot the accuracy of different models as needed

Then, run
```
python accuracyViewer.py
```

## State and Action Representation

### Belief State Representation
We sample each driver's initial location uniformly from a set of 16 possible locations at an intersection. We used a single 145-dim vector to represent the state, concatenating information about each driver's (existence, *x* coordinate, *y* coordinate, heading angle, one-hot vector for its previous action). If a driver's existence was given a value of 0, the other telemetry data points relevant to that vehicle were also given values of 0. The one-hot action representation is described below. The final element in the vector is the current time step of the world.

### Action Representation
We represent an action as a one-hot vector over the action space {Accelerate, Rotate Left, Rotate Right, Brake, Signal}

### Inference Pair Representation
This is a 2-dim vector, indicating two drivers' IDs, *i* and *j*. *i* is inferring *j* (as a lower-level driver) in this pair.

*Results may vary from the original paper due to randomness, but trends across models should be similar.
