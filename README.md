# Neural Amortized Inference for Nested Multi-agent Reasoning

Code for training and testing models from experiments in the paper [Neural Amortized Inference for Nested Multi-agent Reasoning](https://arxiv.org/pdf/2308.11071.pdf). 

## Installation and Running

- Clone the Repository
- There are instructions on how to train and evaluate models for the Construction and Driving experiments described in the paper within the READMEs of the Construction and CARLO folders respectively

## Dataset Overview

For each experiment, we generate rollouts of agents trying to accomplish their goals either in isolation or by interacting with each other. When training each recognition model to approximate belief inference at a certain level *L*, each agent assumes that others within the world are reasoning at a level *L - 1* using our approach and a lower level recognition model, and enumerates all possibile objectives that agent could have when conducting inference. Agents then use their beliefs about others and the ground truth state of the world to decide how to optimally act to accomplish their goals. 

We save the results of doing reasoning via complete enumeration (exact inference) for hundreds to thousands of different rollouts and train a neural network to approximate the results by minimizing the KL divergence between its predictions and the posterior at each timestep saved within the dataset. In the Driving experiment, we perform an additional step of simulating an agent's next action probabilities given the liklihood of having a certain goal, and train a neural network to predict that distribution.

Specifics on the state and action representations for each experiment are in the READMEs of each folder as well.