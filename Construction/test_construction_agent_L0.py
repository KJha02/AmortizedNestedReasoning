import envs.construction
from agents import construction_agent_L0
from envs.construction import Action
import matplotlib.pyplot as plt
import utils
import shutil
import numpy as np
import itertools
import copy
from collections import deque
import os
import pdb
from pathlib import Path
import train_construction_desire_pred
import test_construction_desire_pred
import torch
import time
import argparse
import random
import datetime
import itertools
import numpy as np
import seaborn as sns
import shutil
import utils
import envs.construction
import agents.construction_agent_L0
import agents.construction_agent_L1
import agents.construction_agent_L2
import envs.construction_sample
import copy
import pickle
import pygame


def plot_L0_snapshot(path, path_gui, state, colored_block_utilities_0, is_nn_inference, is_inference, nn_inference, rollout_desire_int):
    action_space = list(envs.construction.Action)

    num_cols = 4
    num_rows = 1
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 5, num_rows * 6), squeeze=False
    )
    axs = axss.flatten()

    # 1 Plot the environment
    ax = axs[0]
    # - Plot state
    screen = state.pyGamePlot(ax)
    Path(path_gui).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(screen, path_gui)
    im = plt.imread(path_gui)
    ax.imshow(im)

    # 2 Plot Online IS+NN inference
    # - Compute x tick labels
    ax = axs[1]

    # num_possible_block_pairs = len(colored_block_utilities_0)
    # final_util_perms = []
    # for i in range(num_possible_block_pairs):
    #     util = [0] * num_possible_block_pairs
    #     util[i] = 100
    # for util in utilities_permutations:
    #     if util not in final_util_perms:
    #         final_util_perms.append(util)
    # x_tick_labels = []
    # for utilities_permutation in final_util_perms:
    #     labels = []
    #     for block_pair, utility in zip(envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities_permutation):
    #         if utility != 0:
    #             block1, block2 = block_pair
    #             pair_name = (envs.construction.block2color[block1], envs.construction.block2color[block2])
    #             labels.append(f"{pair_name}={utility}")
    #     x_tick_labels.append(', '.join(labels))
    block_pair_list = list(envs.construction.ALL_BLOCK_PAIRS[:len(colored_block_utilities_0)])
    block_pair = max(colored_block_utilities_0, key=colored_block_utilities_0.get)
    block_1, block_2 = block_pair
    pair_name = (envs.construction.block2color[block_1], envs.construction.block2color[block_2])
    x_tick_labels = [f"{pair_name}=100"]

    if is_nn_inference is not None:
        # - Plot bars
        if np.argmax(is_nn_inference) == rollout_desire_int:
            bar_color = "red"
        else:
            bar_color = "C0"
        ax.bar(np.arange(len(x_tick_labels)), is_nn_inference[rollout_desire_int], color=bar_color)

        ax.set_title("1's belief about 0's desires using Online IS+NN")
    else:
        ax.set_title("1's belief about 0's desires using Online IS+NN (N/A)")

    # - Formatting
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.tick_params(axis='x', which='both', pad=50)


    # 3 Plot IS Inference
    ax = axs[2]

    if is_inference is not None:
        # - Plot bars
        if np.argmax(is_inference) == rollout_desire_int:
            bar_color = "red"
        else:
            bar_color = "C0"
        ax.bar(np.arange(len(x_tick_labels)), is_inference[rollout_desire_int], color=bar_color)
        ax.set_title("1's belief about 0's desires using IS")
    else:
        ax.set_title("1's belief about 0's desires using IS (N/A)")

    # - Formatting
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.tick_params(axis='x', which='both', pad=50)


    # 4 Plot NN inference
    # - Compute x tick labels
    ax = axs[3]
    
    if nn_inference is not None:
        # - Plot bars
        if np.argmax(nn_inference) == rollout_desire_int:
            bar_color = "red"
        else:
            bar_color = "C0"
        ax.bar(np.arange(len(x_tick_labels)), nn_inference[rollout_desire_int], color=bar_color)

        ax.set_title("1's belief about 0's desires using NN")
    else:
        ax.set_title("1's belief about 0's desires using NN (N/A)")

    # - Formatting
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.tick_params(axis='x', which='both', pad=50)

    # Format figure
    suptitle_kwargs = {"fontsize": 16}
    # colored_block_utilities_L0_list = []
    # for block_pair, utility in colored_block_utilities_0.items():
    #     if utility != 0:
    #         block1, block2 = block_pair
    #         pair_name = (envs.construction.block2color[block1], envs.construction.block2color[block2])
    #         colored_block_utilities_L0_list.append(f"{pair_name}={utility}")
    # colored_block_utilities_L0_str = ', '.join(colored_block_utilities_L0_list)
    fig.suptitle(f"Agent 0 desires: ({pair_name})", **suptitle_kwargs)
    utils.general.save_fig(fig, path, tight_layout_kwargs={"rect": [0, 0, 1, 0.95]})




def evaluate_L0_interaction(env, agent):
    # Interact with the environment
    obs = env.reset()
    done, cumulative_reward, timestep = False, 0, 0
    cumulative_rewards = []
    while not done:
        action, action_info = agent.get_action(obs, return_info=True)
        next_obs, reward, done, info = env.step(action)

        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        assert next_obs == env.transition(obs, action)
        obs = next_obs
        timestep += 1
        print(
            f"t = {timestep} | action = {action} | reward = {reward} | "
            f"cumulative_reward = {cumulative_reward} | done = {done} | agent inventory = {obs.block_picked} | "
            f"block_locations = {obs.colored_blocks} | agent_loc = {obs.agent_location}"
        )

if __name__ == "__main__":
    env = envs.construction.ConstructionEnv(
        initial_state= envs.construction.State(
            gridworld= envs.construction.Gridworld(
                [
                    "***********",
                    "*.........*",
                    "*.........*",
                    "*.........*",
                    "*.....▲...*",
                    "*....x....*",
                    "*.........*",
                    "*.........*",
                    "*.........*",
                    "*.+.....=.*",
                    "***********"
                ]
            ),
            agent_location=(4, 5),
            colored_blocks={
                "x": (5, 5),
                "+": (9, 2),
                "=": (9, 8)
            }
        ),
        colored_block_utilities=envs.construction.get_default_colored_block_utilities(len(envs.construction.ALL_BLOCK_PAIRS))
    )

    beta = 10
    agent = construction_agent_L0.AgentL0(env.state.gridworld, env.colored_block_utilities, env.transition, beta=beta)

    # state = env.state
    # env.step(envs.construction.Action.LEFT)
    # res = env.transition(state, envs.construction.Action.LEFT)
    # print(res)
    evaluate_L0_interaction(env, agent)

