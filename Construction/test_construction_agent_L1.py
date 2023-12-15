import os
import pdb

import test_construction_desire_pred
import time
import argparse
import agents.construction_agent_L1
import envs.construction as construction
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import utils
import random
import shutil
import torch
import train_construction_desire_pred
import envs.construction_sample
import test_reasoning_about_construction_L0

# TODO: merge this with envs.construction.State.plot
def plot_L1_state(gridworld, agent_location_L0, agent_location_L1, ax=None):
    num_rows = gridworld.num_rows
    num_cols = gridworld.num_cols
    if ax is None:
        _, ax = plt.subplots()

    ax.set_axis_off()
    table = matplotlib.table.Table(ax)

    width, height = 1.0 / num_cols, 1.0 / num_rows
    for row in range(len(gridworld.map)):
        for col in range(len(gridworld.map[0])):
            val = gridworld.map[row][col]
            fontproperties = matplotlib.font_manager.FontProperties(family="sans-serif", size="x-large")
            if val == '.':
                facecolor = "white"
                text = ""
            elif val == '*':
                facecolor = "black"
                text = ""
            elif val in construction.ALL_COLORED_BLOCKS:
                facecolor = construction.block2color[val]
                text = construction.block2color[val]

            if agent_location_L0 == (row, col):
                assert val != '*'
                facecolor = "lightblue"
                text = "Agent L0"

            if agent_location_L1 == (row, col):
                assert val != '*'
                facecolor = "red"
                text = "Agent L1"

            if agent_location_L0 == (row, col) and agent_location_L1 == (row, col):
                assert val != '*'
                facecolor = "purple"
                text = "Agents L0 & L1"

            table.add_cell(
                row,
                col,
                width,
                height,
                text=text,
                facecolor=facecolor,
                fontproperties=fontproperties,
                loc="center",
            )

    ax.add_table(table)
    return ax

def get_probs_from_posterior(posterior):
    # Extract num_possible_block_pairs
    utilities, prob = posterior[0]
    num_possible_block_pairs = len(utilities)

    final_util_perms = []
    for i in range(num_possible_block_pairs):
        util = [0] * num_possible_block_pairs
        util[i] = 100
        final_util_perms.append(util)

    probs_order = [
        dict(zip(construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities))
        for utilities in final_util_perms
    ]

    probs = np.zeros((num_possible_block_pairs,))
    for utilities, prob in posterior:
        probs[probs_order.index(dict(utilities))] = prob

    probs /= probs.sum()

    return probs


def plot_L1_snapshot(
    path,
    colored_block_utilities_L0,
    base_colored_block_utilities_L1,
    obs,
    agent_location_L1,
    action_info,
    action,
    cumulative_rewards,
    seek_conflict,
):
    """Plots a snapshot of L1 acting"""
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    # 1 Plot the environment
    ax = axs[0]

    # - Plot StateL0 and the location of agent L1
    plot_L1_state(
        obs.state_L0.gridworld,
        obs.state_L0.agent_location,
        agent_location_L1,
        ax,
    )

    # - Plot the action of agent L0
    ax.set_title(f"\nL0's action: {obs.action_L0.name}")

    # 2 Plot belief
    ax = axs[1]

    # - Compute bars

    # -- Compute x tick labels
    # --- Extract num_possible_block_pairs
    utilities, prob = action_info["belief"][0]
    num_possible_block_pairs = len(utilities)

    # --- xticklabels
    utilities_permutations = list(
        itertools.permutations(construction.ALL_UTILITIES[:num_possible_block_pairs])
    )
    final_util_perms = []
    for util in utilities_permutations:
        if util not in final_util_perms:
            final_util_perms.append(util)
    x_tick_labels = []

    for utilities_permutation in final_util_perms:
        labels = []
        for block_pair, utility in zip(construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities_permutation):
            if utility != 0:
                block1, block2 = block_pair
                pair_name = (construction.block2color[block1], construction.block2color[block2])
                labels.append(f"{pair_name}={utility}")
        x_tick_labels.append(', '.join(labels))


    # --- Compute probs
    probs = get_probs_from_posterior(action_info["belief"])

    # - Plot bars
    ax.bar(np.arange(len(probs)), probs, color="C0")

    # - Formatting
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_title("L1's belief about L0's desires")

    # 3 Plot L1's action probs
    ax = axs[2]

    # - Plot bars
    action_space = list(construction.Action)
    ax.bar(np.arange(len(action_space)), action_info["action_probs"])

    # - Formatting
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(length=0)
    ax.set_xticks(range(len(action_space)))
    ax.set_xticklabels([action.name for action in action_space])
    ax.set_title(f"L1's action probabilities\nL1's sampled action: {action.name}")

    # 4 Plot L1's cumulative reward
    ax = axs[3]

    ax.plot(cumulative_rewards, marker="x")

    # - Formatting
    ax.set_xticks([0, len(cumulative_rewards) - 1])
    ax.set_xlabel("Timestep (0-indexed)")
    ax.set_title("L1's cumulative reward")
    ax.tick_params(direction="in")

    # Format figure
    suptitle_kwargs = {"fontsize": 16}
    colored_block_utilities_L0_list = []
    for block_pair, utility in colored_block_utilities_L0.items():
        if utility != 0:
            block1, block2 = block_pair
            pair_name = (construction.block2color[block1], construction.block2color[block2])
            colored_block_utilities_L0_list.append(f"{pair_name}={utility}")
    colored_block_utilities_L0_str = ', '.join(colored_block_utilities_L0_list)

    base_colored_block_utilities_L1_list = []
    for block_pair, utility in colored_block_utilities_L0.items():
        if utility != 0:
            block1, block2 = block_pair
            pair_name = (construction.block2color[block1], construction.block2color[block2])
            base_colored_block_utilities_L1_list.append(f"{pair_name}={utility}")
    base_colored_block_utilities_L1_str = ', '.join(base_colored_block_utilities_L1_list)

    if seek_conflict:
        fig.suptitle(
            f"L1 agent ({base_colored_block_utilities_L1_str}) seeks conflict with an L0 agent "
            f"({colored_block_utilities_L0_str}) "
            f"(Timestep = {len(cumulative_rewards) - 1})",
            **suptitle_kwargs,
        )
    else:
        fig.suptitle(
            f"L1 agent ({base_colored_block_utilities_L1_str}) does NOT seek conflict with an "
            f"L0 agent ({colored_block_utilities_L0_str}) "
            f"(Timestep = {len(cumulative_rewards) - 1})",
            **suptitle_kwargs,
        )

    utils.general.save_fig(fig, path, tight_layout_kwargs={"rect": [0, 0, 1, 0.95]})

def evaluate_L1_interaction(env, agent, gif_path):
    img_paths = []
    tmp_dir = utils.general.get_tmp_dir()

    # Interact with the environment
    obs = env.reset()
    done, cumulative_reward, timestep = False, 0, 0
    agent_location_L1 = env.agent_location_L1
    cumulative_rewards = []
    while not done:
        if obs is None:
            state = None
            action_L0 = None
        else:
            # state = construction.get_state_L0_with_agent_location_L1_str(
            #     obs.state_L0, agent_location_L1
            # )
            state = obs.state_L0
            action_L0 = obs.action_L0
        try:
            print(f"t = {timestep}\n" f"State = {state}\n" f"L0 action = {action_L0.name}\n")
        except:
            pdb.set_trace()
            env.get_done(agent.agent_location_L1, envs.construction.Action.DOWN)

        # if timestep == 1:
        #     pdb.set_trace()
        try:
            action, action_info = agent.get_action(obs, return_info=True)
        except:
            pdb.set_trace()
            action, action_info = agent.get_action(obs, return_info=True)

        print("L1 belief")
        test_reasoning_about_construction_L0.print_posterior(action_info["belief"])

        action_space = list(construction.Action)
        print(
            "\nAction L1 probs: "
            f"{list(zip(action_space, action_info['action_probs']))}"
            f" | Action L1 = {action.name}"
        )
        next_obs, reward, done, _ = env.step(action)
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        print(
            f"L1 action = {action.name}\n"
            f"L1 reward = {reward}\n"
            f"L1 cumulative_reward = {cumulative_reward}\nL1 done = {done}\n"
        )
        img_path = f"{tmp_dir}/{timestep}.png"
        plot_L1_snapshot(
            img_path,
            env.colored_block_utilities_L0,
            agent.base_colored_block_utilities,
            obs,
            agent_location_L1,
            action_info,
            action,
            cumulative_rewards,
            env.seek_conflict,
        )
        img_paths.append(img_path)

        obs = next_obs
        agent_location_L1 = env.agent_location_L1
        timestep += 1

    # Make gif
    utils.general.make_gif(img_paths, gif_path, 3)
    shutil.rmtree(tmp_dir)

    return cumulative_rewards[-1]

def plot_total_rewardss(total_rewardss, path):
    fig, ax = plt.subplots(1, 1)
    y0s = np.arange(len(total_rewardss))
    height = 0.8
    algorithms = []

    for y0, (algorithm, total_rewards) in zip(y0s, total_rewardss.items()):
        algorithms.append(algorithm)

        # Plot histogram and individual total rewards
        test_construction_desire_pred.plot_hist(total_rewards, y0, height, ax)
        ax.scatter(
            total_rewards,
            [y0 + 0.1 for _ in range(len(total_rewards))],
            s=10.0,
            color="black",
            zorder=100,
        )

        # Annotate each individual run
        for i, total_reward in enumerate(total_rewards):
            ax.text(
                total_reward,
                y0 + 0.2,
                f"{i}",
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
                fontsize=3,
                zorder=100,
            )

        # Draw a mean line
        ax.plot([np.mean(total_rewards) for _ in range(2)], [y0, y0 + 1], color="C0")

        # Separate different algorithms
        ax.axhline(y0, color="C0")

    # Format figure
    ax.set_yticks(y0s)
    ax.set_yticklabels(algorithms)
    ax.set_ylim(0, y0s[-1] + 1)
    ax.tick_params(length=0, axis="y")
    ax.set_ylabel("Algorithm")

    ax.set_xlabel("Total reward")
    ax.set_title("Distribution of total rewards of an L1 agent")
    utils.general.save_fig(fig, path)

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    plotted_something = False
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            continue

        # Load model
        model, optimizer, stats, train_args = train_construction_desire_pred.load_checkpoint(
            checkpoint_path, device
        )
        model.eval()
        config_name = train_construction_desire_pred.get_config_name(train_args)

        algorithms = [
            # "NN",
            # "SMC",
            # "SMC+NN",
            "IS",
            # "IS+NN",
            # "SMC+NN+rejuvenation",
            # "Online_IS+NN",
            # "oracle",
            # "baseline",
            "SMC(100)",
        ]
        total_rewardss = {algorithm: [] for algorithm in algorithms}
        for i in range(args.num_environments):
            # Create env
            env = envs.construction_sample.sample_construction_env_L1(
                train_args.num_colored_block_locations, train_args.num_possible_block_pairs
            )
            for algorithm in algorithms:
                env.reset()

                # pdb.set_trace()

                # Create agent
                agent = agents.construction_agent_L1.AgentL1(
                    env.seek_conflict,
                    env.base_colored_block_utilities_L1,
                    train_args.num_possible_block_pairs,
                    env.initial_state_L0,
                    env.agent_location_L1,
                    env.env_L0.transition,
                    env.transition,
                    algorithm,
                    num_samples=5,
                    model=model,
                    ground_truth_colored_block_utilities_L0=env.colored_block_utilities_L0,
                )
                # Interact with the environment
                total_rewardss[algorithm].append(
                    evaluate_L1_interaction(
                        env,
                        agent,
                        f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                        f"{config_name}/L1/interaction_gifs/{i}/{algorithm}.gif",
                    )
                )

            plot_total_rewardss(
                total_rewardss,
                f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                f"{config_name}/L1/total_rewards.pdf",
            )
        plotted_something = True

    return plotted_something

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--num-environments", type=int, default=20, help="Number of environments to test"
    )
    parser.add_argument("--env-name", type=str, default="construction", help="Environment name")
    parser.add_argument("--save-dir", type=str, default="save", help="Save directory")
    parser.add_argument("--experiment-name", type=str, default="", help=" ")
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
