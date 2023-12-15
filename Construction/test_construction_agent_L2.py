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
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import utils
import envs.construction
import agents.construction_agent_L0
import agents.construction_agent_L1
import agents.construction_agent_L2
import envs.construction_sample
import utils.construction_data
import copy
import test_construction_agent_L1
import pickle
import pygame

def plot_L2_snapshot(
    path, path_gui, state, colored_block_utilities_0, action_1, action_1_info, action_0, action_0_info, L1_conflict_str=None
):
    """Plots a snapshot of L2 acting

    Args
        path (str)
        state (StateMultiAgent)
    """
    action_space = list(envs.construction.Action)
    num_cols = 1
    num_rows = 1
    # num_cols = 4
    # num_rows = 2
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 5, num_rows * 6), squeeze=False
    )
    axs = axss.flatten()

    # 1 Plot the environment
    ax = axs[0]
    # - Plot state
    screen = state.plot(ax)
    Path(path_gui).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(screen, path_gui)
    im = plt.imread(path_gui)
    ax.imshow(im)

    # # 2 Plot belief of L0's goals
    # #ax = axs[7]
    # ax = axs[1]

    # # - Compute x tick labels
    # if colored_block_utilities_0 is not None:
    #     num_possible_block_pairs = len(colored_block_utilities_0)
    # else:
    #     num_possible_block_pairs = 45
    # final_util_perms = []
    # for i in range(num_possible_block_pairs):
    #     util = [0] * num_possible_block_pairs
    #     util[i] = 100
    #     final_util_perms.append(util)
    # x_tick_labels = []
    # for utilities_permutation in final_util_perms:
    #     labels = []
    #     for block_pair, utility in zip(envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities_permutation):
    #         if utility != 0:
    #             block1, block2 = block_pair
    #             pair_name = (envs.construction.block2color[block1], envs.construction.block2color[block2])
    #             labels.append(f"{pair_name}")
    #     x_tick_labels.append(''.join(labels))

    # labelsToUse = []
    # correctL0UtilityInt = utils.construction_data.block_pair_utilities_to_desire_int(colored_block_utilities_0, 45)
    # if action_1_info is not None:
    #     # - Compute bars TODO: merge with test_agent_L1.plot_L1_snapshot
    #     utilities, prob = action_1_info["belief"][0]
    #     probs = test_construction_agent_L1.get_probs_from_posterior(action_1_info["belief"])
    #     probs = list(probs)
    #     # - Plot bars of 5 highest probabilities
    #     probsToPlot = sorted(probs, reverse=True)
    #     bar_color = "C0"
    #     if probs.index(probsToPlot[0]) == correctL0UtilityInt:
    #         bar_color = 'green'

    #     temp = []
    #     for i in range(5):
    #         temp.append(probsToPlot[i])
    #         lIdx = probs.index(probsToPlot[i])
    #         label = x_tick_labels[lIdx]
    #         labelsToUse.append(label)
    #         probs.pop(lIdx) # do this to avoid duplicate items with max
    #         x_tick_labels.pop(lIdx)
        


    #     for ele in labelsToUse:
    #         try:
    #             assert labelsToUse.count(ele) == 1
    #         except:
    #             pdb.set_trace()



    #     ax.bar(np.arange(len(temp)), temp, color=bar_color)

    #     ax.set_title("Bob's belief about Alice's desires")
    # else:
    #     ax.set_title("Bob's belief about Alice's desires (N/A)")

    # # - Formatting
    # ax.set_ylim(0, 1)
    # ax.set_yticks([0, 1])
    # ax.tick_params(length=0)
    # ax.tick_params(axis="x", labelrotation=90)
    # ax.set_xticks(range(len(labelsToUse)))
    # ax.set_xticklabels(labelsToUse)
    # ax.tick_params(axis='x', which='both', pad=50)

    # # 3 Plot L2's Guess about L1's intentions
    # ax = axs[2]
    # helper_x_tick_labels = ["Helping", "Hurting"]
    # if action_0_info is not None:
    #     prob_dict = action_0_info["other_agent_seek_conflict"]
    #     probs = []
    #     if type(prob_dict) != dict:
    #         probs.append(prob_dict[0])
    #         probs.append(prob_dict[1])
    #         prob_dict = {False: prob_dict[0], True: prob_dict[1]}

    #     else:
    #         if False in prob_dict:
    #             probs.append(prob_dict[False])
    #         else:
    #             probs.append(1 - prob_dict[True])
    #         if True in prob_dict:
    #             probs.append(prob_dict[True])
    #         else:
    #             probs.append(1 - prob_dict[False])

    #     # - Plot bars
    #     bar_color = "C0"
    #     if len(action_0_info["prev_other_agent_seek_conflict"]) > 0:
    #         #prev_prob_dict = action_0_info["prev_other_agent_seek_conflict"][-1]
    #         #max_prev_prob = max(prev_prob_dict, key=prev_prob_dict.get)
    #         max_prob = max(prob_dict, key=prob_dict.get)
    #         #if max_prob != max_prev_prob:
    #         if max_prob == action_1_info['conflict']:
    #             bar_color = "green"
    #     ax.bar(np.arange(len(probs)), probs, color=bar_color)
    #     ax.set_title("Alice's belief about Bob's friendliness")
    # else:
    #     ax.set_title("Alice's belief about Bob's friendliness (N/A)")
    # ax.set_ylim(0, 1)
    # ax.set_yticks([0, 1])
    # ax.tick_params(length=0)
    # ax.tick_params(axis="x", labelrotation=90)
    # ax.set_xticks(range(len(helper_x_tick_labels)))
    # ax.set_xticklabels(helper_x_tick_labels)

    # # 4 Plot 1's action probs
    # ax = axs[1]

    # # - Plot bars
    # if action_1_info is not None:
    #     ax.bar(np.arange(len(action_space)), action_1_info["action_probs"])
    # else:
    #     action_probs = np.zeros((len(action_space),))
    #     action_probs[envs.construction.Action.STOP.value] = 1
    #     ax.bar(np.arange(len(action_space)), action_probs)

    # # - Formatting
    # ax.set_ylim(0, 1)
    # ax.set_yticks([0, 1])
    # ax.tick_params(length=0)
    # ax.set_xticks(range(len(action_space)))
    # ax.set_xticklabels([action.name for action in action_space])
    # ax.set_title(f"1's action probabilities (1's action: {action_1.name})")

    # # 5 Plot 0's belief about 1's belief
    # ax = axs[6]
    # other_agent_current_belief = action_0_info["other_agent_current_belief"]

    # if other_agent_current_belief is not None:
    #     # - Compute bars TODO: merge with test_agent_L1.plot_L1_snapshot
    #     utilities, prob = other_agent_current_belief[0]
    #     probs = test_construction_agent_L1.get_probs_from_posterior(other_agent_current_belief)

    #     # - Plot bars
    #     ax.bar(np.arange(len(probs)), probs, color="C1")

    #     ax.set_title("0's knowledge about\n1's belief about 0's desires")
    # else:
    #     ax.set_title("0's knowledge about\n1's belief about 0's desires (N/A)")

    # # - Formatting
    # ax.set_ylim(0, 1)
    # ax.set_yticks([0, 1])
    # ax.tick_params(length=0)
    # ax.tick_params(axis="x", labelrotation=90)
    # ax.set_xticks(range(len(x_tick_labels)))
    # ax.set_xticklabels(x_tick_labels)
    # ax.tick_params(axis='x', which='both', pad=50)

    # # 6 Plot 0's action probs
    # ax = axs[3]

    # # - Plot bars
    # action_space = list(envs.construction.Action)
    # try:
    #     ax.bar(
    #         np.arange(len(action_space)),
    #         [action_0_info["L2_action_probs"][action] for action in action_space],
    #         color="C1",
    #     )
    # except:
    #     ax.bar(
    #         np.arange(len(action_space)),
    #         np.zeros((len(action_space),)),
    #         color="C1",
    #     )

    # # - Formatting
    # ax.set_ylim(0, 1)
    # ax.set_yticks([0, 1])
    # ax.tick_params(length=0)
    # ax.set_xticks(range(len(action_space)))
    # ax.set_xticklabels([action.name for action in action_space])
    # action_0_name = agents.agent_L2.NOOP if action_0 == agents.agent_L2.NOOP else action_0.name
    # ax.set_title(
    #     "L2 action probabilities"
    #     f" (0's action: {action_0_name})"
    # )

    # # 7 Plot 1's next beliefs given 0's actions
    # imagined_probs = action_0_info["L1_imagined_probs"]
    # imagined_probs = imagined_probs[::-1]  # reverse order of actions
    # social_goals = [False, True]
    # for conflict_id, conflict in enumerate(social_goals):
    #     ax = axs[4 + conflict_id]
    #     probs = imagined_probs[conflict_id]
    #     ax.bar(np.arange(len(probs)), probs, color="C2")
    #     if not conflict:
    #         ax.set_title(f"L2's beliefs about L1's action probabilities\nif L1 is helping")
    #     else:
    #         ax.set_title(f"L2's beliefs about L1's action probabilities\nif L1 is hurting")

    #     # - Formatting
    #     ax.set_ylim(0, 1)
    #     ax.set_yticks([0, 1])
    #     ax.tick_params(length=0)
    #     ax.tick_params(axis="x", labelrotation=90)
    #     ax.set_xticks(range(len(action_space)))
    #     ax.set_xticklabels([action.name for action in action_space])

    #     if action_1_info is not None:
    #         L1_prev_act = action_1_info["prev_L1_action"]
    #         ax.get_children()[L1_prev_act.value].set_color("black")  # set ground truth bar to black

    # Format figure
    # suptitle_kwargs = {"fontsize": 16}
    # colored_block_utilities_L0_list = []

    # for block_pair, utility in colored_block_utilities_0.items():
    #     if utility != 0:
    #         block1, block2 = block_pair
    #         pair_name = (envs.construction.block2color[block1], envs.construction.block2color[block2])
    #         colored_block_utilities_L0_list.append(f"{pair_name}")
    # colored_block_utilities_L0_str = ', '.join(colored_block_utilities_L0_list)
    # if action_1_info is not None:
    #     if not action_1_info['conflict']:
    #         social_goal = "Help"
    #     else:
    #         social_goal = "Hinder"
    #     fig.suptitle(f"Alice's desires: ({colored_block_utilities_L0_str}) \n"
    #                  f" Bob's social goal: {social_goal}", **suptitle_kwargs)
    # elif L1_conflict_str is not None:
    #     fig.suptitle(f"Alice's desires: (N/a) \n"
    #                  f" Agent 1 seek conflict: {L1_conflict_str}", **suptitle_kwargs)
    # else:
    #     fig.suptitle(f"Alice's desires: ({colored_block_utilities_L0_str})", **suptitle_kwargs)

    utils.general.save_fig(fig, path, tight_layout_kwargs={"rect": [0, 0, 1, 1]})

def evaluate_L2_interaction(env_multi_agent, agent_0, agent_1, gif_path):
    """Returns whether agent 0 deceived agent 1"""
    num_agents = env_multi_agent.num_agents
    img_paths = []
    img_gui_paths = []
    tmp_dir = utils.general.get_tmp_dir()

    obs = env_multi_agent.reset()
    done, cumulative_reward, timestep = False, [0, 0], 0
    num_correct_guesses = 0
    while not done and timestep < 40:
        # if timestep == 8:
        #     pdb.set_trace()
        state_copy = pickle.loads(pickle.dumps(env_multi_agent.state))
        # agent_L1_loc = obs['current_agent_locations'][1]
        # grid = pickle.loads(pickle.dumps(obs['gridworld']))
        # map = grid.map
        # for cells in grid.get_second_agent():
        #     envs.construction.strModify(map, cells[0], cells[1], '.')  # removing second agent duplicates
        # envs.construction.strModify(map, agent_L1_loc[0], agent_L1_loc[1], '▲')  # drawing second agent in as a wall

        # Make observation for agent 0
        state_0 = obs['agent_0_observation']
        prev_action_0 = obs["prev_actions"][0]
        state_1 = obs['agent_1_observation']
        prev_action_1 = obs["prev_actions"][1]
        # Step the environments for the agents
        agent_0.other_agent.curr_state_L0 = state_0
        agent_0.other_agent.curr_state_L1 = state_1
        agent_1.curr_state_L0 = state_0
        agent_1.curr_state_L1 = state_1

        # pdb.set_trace()
        action_0, action_0_info = agent_0.get_action(
            my_current_state=state_0, my_previous_action=prev_action_0, return_info=True,
            current_state_L1=state_1, prev_action_L1=prev_action_1
        )
        if action_0_info["L2_guess_conflict"] == env_multi_agent.seek_conflict:
            num_correct_guesses += 1

        #print(state_copy)
        #print(f"Ground truth L0: {env_multi_agent.colored_block_utilities[0]}")
        # for (act, belief) in action_0_info['next_beliefs'].items():
        #     #print(f"Action: {act}")
        #     for beliefs in belief:
        #         pass
                # print(f"P(g | a): {beliefs}")
            #print("\n")
        # pdb.set_trace()
        # Make observation for agent 1
        if prev_action_0 is None:
            observation_1 = None
        else:
            if prev_action_0 == agents.construction_agent_L2.NOOP:
                observation_1 = None
            else:
                # agent_L1_loc = obs['current_agent_locations'][1]
                # map = grid.map
                # for cells in grid.get_second_agent():
                #     envs.construction.strModify(map, cells[0], cells[1], '.')  # removing second agent duplicates
                # envs.construction.strModify(map, agent_L1_loc[0], agent_L1_loc[1], '▲')  # drawing second agent in as a wall
                observation_1 = envs.construction.ObservationL1(state_0, prev_action_0)
                # try:
                #     assert observation_1 == agent_0.other_agent.observations[-1]
                # except:
                #     pdb.set_trace()
        # if timestep == 5 or timestep == 4:
        #     pdb.set_trace()
        # pdb.set_trace()
        action_1, action_1_info = agent_1.get_action(observation_1, return_info=True)

        # Build action
        action = {0: action_0, 1: action_1}
        # if timestep == 0:
        #     pdb.set_trace()
        obs, reward, done, info = env_multi_agent.step(action)
        for agent_id in range(num_agents):
            cumulative_reward[agent_id] += reward[agent_id]
        img_path = f"{tmp_dir}/{timestep}.png"
        img_path_gui = f"{tmp_dir}/{timestep}_gui.png"
        plot_L2_snapshot(
            img_path,
            img_path_gui,
            state_copy,
            env_multi_agent.colored_block_utilities[0],
            action_1,
            action_1_info,
            action_0,
            action_0_info
        )
        img_paths.append(img_path)
        img_gui_paths.append(img_path_gui)
        timestep += 1
        print(
            f"t = {timestep} | action = {action} | reward = {reward} | "
            f"cumulative_reward = {cumulative_reward} | done = {done}"
        )
        print(
            f"inventory = {env_multi_agent.state.agent_inv} | block_locations = {env_multi_agent.state.colored_blocks}"
        )

    # Make gif
    utils.general.make_gif(img_paths, gif_path, 3)
    shutil.rmtree(tmp_dir)

    # Return whether agent 0 deceived agent 1, return average correct guesses of seek_conflict
    return cumulative_reward[0], num_correct_guesses / timestep, timestep


def plot_deception_stats(deceived_ids, num_environments, path):
    fig, ax = plt.subplots(1, 1)
    algorithms = list(deceived_ids.keys())

    ax.bar(
        np.arange(len(algorithms)),
        [len(deceived_ids[algorithm]) for algorithm in algorithms],
        color="C0",
    )
    ax.text(
        0.05,
        0.95,
        "\n".join([f"{algorithm}: {deceived_ids[algorithm]}" for algorithm in algorithms]),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    ax.set_ylim(0, num_environments)
    ax.set_yticks([0, num_environments])
    ax.tick_params(length=0)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms)
    ax.set_title("L1's belief about L0's desires")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Number of deceptions")
    ax.set_title("How many times did L2 deceive L1?")
    utils.general.save_fig(fig, path)


def main(args):
    # Set seed and device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        device = "cuda"
    else:
        device = "cpu"

    # Checkpoint paths
    if args.checkpoint_path is None:
        checkpoint_paths = list(
            test_construction_desire_pred.get_checkpoint_paths(
                args.save_dir, args.env_name, args.experiment_name
            )
        )
    else:
        checkpoint_paths = [args.checkpoint_path]

    # Loop through the models
    plotted_something = False
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            continue

        # # Load model
        # model, optimizer, stats, train_args = train_construction_desire_pred.load_checkpoint(
        #     checkpoint_path, device
        # )
        # model.eval()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        train_args = checkpoint["args"]
        config_name = train_construction_desire_pred.get_config_name(train_args)

        algorithms = [
            # "NN",
            "IS",
            # "SMC",
            # "Online_IS+NN",
            # "SMC(100)",
        ]
        print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))
        deceived_ids = {algorithm: [] for algorithm in algorithms}
        for environment_id in range(args.num_environments):
            # Create environment
            seek_conflict = True
            # env_multi_agent = envs.construction_sample.sample_multi_agent_env(
            #     train_args.num_colored_block_locations, train_args.num_possible_block_pairs, seek_conflict=seek_ conflict
            # )
            conflict_prior = [0]
            comb_weights = [0]

            for seek_conflict in [True, False]:
                env_multi_agent = envs.construction_sample.default_multi_agent_env(seek_conflict)
                heatmap_data_conflict = []
                heatmap_data_speed = []
                print(f"The Ground Truth for L1 is {seek_conflict}")
                for prior in conflict_prior:
                    pct_correct_prior = []
                    num_timestepss = []
                    for comb_weight in comb_weights:
                        for other_agent_inference_algorithm in algorithms:
                            env_multi_agent.reset()
                            # Create agents
                            # - Agent 1 inference
                            agent_1_inference_algorithm = other_agent_inference_algorithm
                            agent_1_num_samples = 3
                            other_agent_num_samples = 3

                            # - Stochastic vs deterministic actions (-10 = completely random, 10 = deterministic)
                            beta_0 = 0.01
                            beta_1 = 0.01

                            # - Initial state
                            # agent_L1_loc = env_multi_agent.initial_state.agent_locations[1]
                            grid = env_multi_agent.initial_state.gridworld
                            # map = grid.map
                            # envs.construction.strModify(map, agent_L1_loc[0], agent_L1_loc[1], '▲')  # drawing second agent in as a wall
                            initial_state_L0 = envs.construction.State(
                                grid,
                                env_multi_agent.initial_state.agent_locations[0],
                                pickle.loads(pickle.dumps(env_multi_agent.initial_state.colored_blocks)),
                            )

                            # - Create transitions
                            transition_L0 = envs.construction.ConstructionEnv(
                                initial_state_L0, env_multi_agent.colored_block_utilities[0]
                            ).transition
                            transition_L1 = envs.construction.ConstructionEnvL1(
                                seek_conflict,
                                env_multi_agent.colored_block_utilities[1],
                                initial_state_L0,
                                env_multi_agent.colored_block_utilities[0],
                                agent_location_L1=env_multi_agent.initial_state.agent_locations[1],
                                agent_inv_L1=env_multi_agent.initial_state.agent_inv[1]
                            ).transition
                            # Create agent 1
                            agent_1 = agents.construction_agent_L1.AgentL1(
                                seek_conflict,
                                env_multi_agent.colored_block_utilities[1],
                                env_multi_agent.num_possible_block_pairs,
                                initial_state_L0,
                                env_multi_agent.initial_state.agent_locations[1],
                                transition_L0,
                                transition_L1,
                                inference_algorithm=agent_1_inference_algorithm,
                                beta_L0=beta_0,
                                beta_L1=beta_1,
                                num_samples=agent_1_num_samples,
                                ground_truth_colored_block_utilities_L0=env_multi_agent.colored_block_utilities[0]
                            )
                            # Create agent 0
                            agent_0 = agents.construction_agent_L2.AgentL2(
                                env_multi_agent.colored_block_utilities[0],
                                initial_state_L0,
                                env_multi_agent.initial_state.agent_locations[1],
                                transition_L0,
                                transition_L1,
                                other_agent_inference_algorithm,
                                other_agent_beta_L0=beta_0,
                                other_agent_beta_L1=beta_1,
                                other_agent_num_samples=other_agent_num_samples,
                                other_agent_model=None,
                                beta_L2=beta_0,
                                prior=prior,
                                comb_weight=comb_weight
                            )
                            # Interact with the environment
                            gif_path = (
                                f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                                f"{config_name}/L2/interaction_gifs/{environment_id}/"
                                f"{other_agent_inference_algorithm}_{seek_conflict}_{prior * 100}_{comb_weight}.gif"
                            )
                            deceived, pct_correct, num_timesteps = evaluate_L2_interaction(env_multi_agent, agent_0, agent_1, gif_path)
                            pct_correct_prior.append(pct_correct)
                            num_timestepss.append(9 - num_timesteps)
                            if deceived:
                                deceived_ids[other_agent_inference_algorithm].append(environment_id)
                            print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))
                            break
                        plot_deception_stats(
                            deceived_ids,
                            environment_id + 1,
                            f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                            f"{config_name}/L2/deception_stats_{seek_conflict}.pdf",
                        )

                    heatmap_data_speed.append(num_timestepss)
                    heatmap_data_conflict.append(pct_correct_prior)


                # TODO: Make heatmap here
                ax = sns.heatmap(heatmap_data_conflict)
                heatmap_path = (
                    f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                    f"{config_name}/L2/heatmap_{seek_conflict}_default.png"
                )
                plt.title("Heatmap of social goal inferences given conflict priors and combination weights")
                plt.xlabel("Conflict Priors", fontsize=15)
                plt.ylabel("Combination Weights", fontsize=15)
                plt.savefig(heatmap_path)

                ax2 = sns.heatmap(heatmap_data_speed)
                speed_hmap_path = (
                    f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                    f"{config_name}/L2/heatmap_speed.png"
                )
                plt.title("Heatmap of difference between L0 speed and L1 helping speed \n given conflict priors and combination weights")
                plt.xlabel("Conflict Priors", fontsize=15)
                plt.ylabel("Combination Weights", fontsize=15)
                plt.savefig(speed_hmap_path)



            break
        plotted_something = True

    return plotted_something


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--num-environments", type=int, default=10, help="Number of environments to test"
    )
    parser.add_argument("--env-name", type=str, default="construction", help="Environment name")
    parser.add_argument("--save-dir", type=str, default="save", help="Save directory")
    parser.add_argument("--experiment-name", type=str, default="construction", help=" ")
    parser.add_argument("--repeat", action="store_true", help="")
    parser.add_argument("--checkpoint-path", type=str, default=None, help=" ")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with torch.no_grad():
        if args.repeat:
            while True:
                plotted_something = main(args)
                if not plotted_something:
                    print("Didn't plot anything ... waiting 30 seconds")
                    time.sleep(30)
        else:
            main(args)
