import argparse
import copy
import pdb
import torch
import random
from pathlib import Path
import _pickle as pickle
import utils
import pdb
import matplotlib.pyplot as plt
import numpy as np
import itertools
import envs.construction
import utils.construction_data
import utils.general
import shutil
import time
import train_construction_desire_pred
import os
import test_reasoning_about_construction_L0
import scipy.stats
from envs.construction import block2color
import models
import torch.optim as optim

def init(args, device):
    if args.utility_mode == "ranking":
        output_dim = envs.construction.get_num_rankings(args.num_colored_block_locations)
    elif args.utility_mode == "top":
        if args.num_possible_block_pairs != \
                envs.construction.get_num_rankings(args.num_colored_block_locations):
            raise ValueError(
                "Number of possible block pairs must be equal to number of colored block "
                "locations choose 2 in 'top' utility mode"
            )
        output_dim = args.num_possible_block_pairs
    # print("Output dimensions is:" + str(output_dim))
    if args.model_type == "ToMnet_DesirePred":
        model = models.ToMnet_DesirePred(
            state_dim=(2*args.num_colored_block_locations + 4,args.num_rows, args.num_cols, ),
            action_size=args.action_size,
            num_channels=args.num_channels,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            rank=0
        )
    else:
        raise ValueError("Invalid model type:", args.model_type)
    model.to(device)
    # pred_model = DDP(model, device_ids=[device], output_device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    return model, optimizer


def init_actionPred(args, device):
    if args.utility_mode == "ranking":
        utility_dim= envs.construction.get_num_rankings(args.num_colored_block_locations)
    elif args.utility_mode == "top":
        if args.num_possible_block_pairs != \
                envs.construction.get_num_rankings(args.num_colored_block_locations):
            raise ValueError(
                "Number of possible block pairs must be equal to number of colored block "
                "locations choose 2 in 'top' utility mode"
            )
        utility_dim = args.num_possible_block_pairs
    action_model = models.ToMnet_DesirePred(
            state_dim=(3*args.num_colored_block_locations + 4,args.num_rows, args.num_cols, ),
            action_size=args.action_size,
            num_channels=args.num_channels,
            hidden_dim=args.hidden_dim,
            output_dim=2,
        )
    action_model.to(device)
    action_optimizer = optim.Adam(action_model.parameters(), lr=args.lr)
    return action_model, action_optimizer


def save_checkpoint(path, model, optimizer, stats, args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if model.rank == 0:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats,
                "args": args,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

def load_checkpoint(path, device, num_tries=3, L1=False):
    for i in range(num_tries):
        try:
            checkpoint = torch.load(path, map_location=device)
            break
        except Exception as e:
            print(f"Error {e}")
            wait_time = 2 ** i
            print(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
    args = checkpoint["args"]
    if L1:
        model, optimizer = init_actionPred(args, device)
    else:
        model, optimizer = init(args, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, args


def plot_kde(data, y0, height, ax, color="C0"):
    """https://stackoverflow.com/a/46647714"""
    data_range = data.max() - data.min()
    margin = 0.5
    x = np.linspace(data.min() - margin * data_range, data.max() + margin * data_range)
    y = scipy.stats.gaussian_kde(data)(x)
    ax.plot(x, y0 + y / y.max() * height, color=color)
    ax.fill_between(x, y0 + y / y.max() * height, y0, color=color, alpha=0.2)


def plot_hist(data, y0, height, ax, color="C0"):
    """https://stackoverflow.com/a/53528956"""
    n, bins, rects = ax.hist(data, bins="auto", bottom=y0, histtype="bar", color=color, alpha=0.5)

    # adjust heights
    max_height = max([r.get_height() for r in rects])
    for r in rects:
        r.set_height(r.get_height() / max_height * height)


def get_kl(p, q):
    """
    KL divergence between two probability vectors

    Args
        p (numpy.ndarray [*shape, event_dim])
        q (numpy.ndarray [*shape, event_dim])

    Returns [*shape]
    """
    return torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=torch.from_numpy(p)),
        torch.distributions.Categorical(probs=torch.from_numpy(q)),
    ).numpy()


def get_l2(p, q):
    """
    L2 distance between two probability vectors

    Args
        p (numpy.ndarray [*shape, event_dim])
        q (numpy.ndarray [*shape, event_dim])

    Returns [*shape]
    """
    return np.linalg.norm(p - q, axis=-1)


def get_nn_probs(model, states, actions):
    """Get probabilities of desires at every timestep
    from a neural network model ran on a state-action trajectory.

    Args
        model (ToMNet.ToMnet_DesirePred instance)
        states: tensor [num_timesteps, num_rows, num_cols, 2*num_possible_colored_blocks + 4]
            states[t, r, c] is a one hot where
                d=0 -- nothing
                1 -- wall
                2 -- second agent
                3 to (3 + num_colored_blocks - 1) -- colored block type
                3 + num_possible_colored_blocks -- agent
                3 + num_colored_blocks to 3 + 2 * num_colored_blocks -- colored block in inventory
        actions: tensor [num_timesteps, num_actions=5] - one hot representation
            the first action is always [0, 0, 0, 0, 0] -- corresponding to no action
            [1, 0, 0, 0, 0] ... [0, 0, 0, 0, 1] correpond to UP, DOWN, LEFT, RIGHT, PUT_DOWN
            based on envs.construction.Action

    Returns
        probs (np.ndarray [num_timesteps, num_possible_rankings = num_possible_food_trucks!]
                       or [num_timesteps, num_possible_food_trucks])
            where probs[time] is a probability vector
    """
    last = 0
    # Extract values
    _, num_rows, num_cols, _ = states.shape
    action_size = actions.shape[-1]
    device = actions.device

    # NN predictions
    lens = torch.LongTensor([states.shape[0]]).to('cpu')
    actions_2d = [utils.network.expand_batch(actions, (num_rows, num_cols, action_size))]

    log_prob = model([states], actions_2d, lens, last=last)
    probs = torch.softmax(log_prob, 1).cpu().detach().numpy()
    probs /= probs.sum(axis=1, keepdims=True)
    return probs

def get_prob_from_inference_output(inference_output, num_timesteps, num_possible_block_pairs):
    utilities_permutations = list(
        itertools.permutations(envs.construction.ALL_UTILITIES[:num_possible_block_pairs])
    )
    final_util_perms = []
    for util in utilities_permutations:
        if util not in final_util_perms:
            final_util_perms.append(util)
    probs_order = [
        dict(zip(envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities))
        for utilities in final_util_perms
    ]

    probs = np.zeros((num_timesteps, num_possible_block_pairs))
    for timestep, (samples, log_weights) in enumerate(inference_output):
        posterior = test_reasoning_about_construction_L0.get_posterior(samples, log_weights)
        for utilities, prob in posterior:
            probs[timestep, probs_order.index(dict(utilities))] = prob

    probs /= probs.sum(axis=1, keepdims=True)

    return probs

def get_particle_inference_probs(states, actions, desires, algorithm, num_samples=100, model=None):
    """Get probabilities of desires at every timestep
    from IS/SMC ran on a state-action trajectory.
    If model is supplied, use neural network proposal.

    Args
        states: tensor [num_timesteps, num_rows, num_cols, 2*num_possible_colored_blocks + 4]
            states[t, r, c] is a one hot where
                d=0 -- nothing
                1 -- wall
                2 -- second agent
                3 to (3 + num_colored_blocks - 1) -- colored block type
                3 + num_possible_colored_blocks -- agent
                3 + num_colored_blocks to 3 + 2 * num_colored_blocks -- colored block in inventory
        actions: tensor [num_timesteps, num_actions=6] - one hot representation
            the first action is always [0, 0, 0, 0, 0, 1] -- corresponding to no action
            [1, 0, 0, 0, 0, 0] ... [0, 0, 0, 0, 1] correpond to UP, DOWN, LEFT, RIGHT, PUT_DOWN, STOP
            based on envs.construction.Action
        desires: tensor [num_timesteps]
        algorithm (str): one of ["NN", "SMC", "SMC+NN", "IS", "IS+NN",
                                 "SMC+NN+rejuvenation", "SMC(100)","Online_IS+NN"]
        num_samples (int; default 100)
        model: None or ToMNet.ToMnet_DesirePred instance, in which case it is used as
            a proposal distribution

    Returns
        probs (np.ndarray [num_timesteps, num_possible_rankings = num_possible_food_trucks!])
            where probs[time] is a probability vector
    """
        # Extract values
    num_timesteps = states.shape[0]
    num_possible_colored_blocks = int((states.shape[-1] - 4) / 2)
    num_possible_block_pairs = int(num_possible_colored_blocks * (num_possible_colored_blocks - 1) / 2)
    utilities = utils.construction_data.desire_int_to_utilities(
        desires[0].item(), num_possible_block_pairs
    )  # TODO why do we need this during inference
    states_ = [utils.construction_data.state_tensor_to_state(state) for state in states]
    actions_ = [utils.construction_data.action_tensor_to_action(action) for action in actions]
    env = envs.construction.ConstructionEnv(initial_state=states_[0], colored_block_utilities=utilities)
    if algorithm == "NN":
        return get_nn_probs(model, states, actions)
    elif algorithm == "Online_IS+NN":
        env_ = pickle.loads(pickle.dumps(env))
        inference_output = test_reasoning_about_construction_L0.online_importance_sampling(
            env_,
            states_,
            actions_,
            num_samples,
            colored_block_utilities_proposal_probss=get_nn_probs(model, states, actions),
        )
        return get_prob_from_inference_output(
            inference_output, num_timesteps, num_possible_block_pairs
        )
    else:
        if algorithm == "SMC":
            colored_block_utilities_proposal_probss = None
            resample = True
            rejuvenate = False
        elif algorithm == "SMC(100)":
            colored_block_utilities_proposal_probss = None
            resample = True
            rejuvenate = False
            num_samples = 100
        elif algorithm == "SMC+NN":
            colored_block_utilities_proposal_probss = get_nn_probs(model, states, actions)
            resample = True
            rejuvenate = False
        elif algorithm == "SMC+NN+rejuvenation":
            colored_block_utilities_proposal_probss = get_nn_probs(model, states, actions)
            resample = True
            rejuvenate = True
        elif algorithm == "IS":
            colored_block_utilities_proposal_probss = None
            resample = False
            rejuvenate = False
        elif algorithm == "IS+NN":
            colored_block_utilities_proposal_probss = get_nn_probs(model, states, actions)
            resample = False
            rejuvenate = False
        else:
            raise NotImplementedError(f"{algorithm} not implemented yet")
        env_ = pickle.loads(pickle.dumps(env))
        inference_output = test_reasoning_about_construction_L0.particle_inference(
            env_,
            states_,
            actions_,
            resample=resample,
            rejuvenate=rejuvenate,
            num_samples=num_samples,
            colored_block_utilities_proposal_probss=colored_block_utilities_proposal_probss,
            output_every_timestep=True,
        )
        return get_prob_from_inference_output(
            inference_output, num_timesteps, num_possible_block_pairs
        )

def plot_reasoning_about_L0(
    posterior_error_path, posterior_gifs_dir, dataset, model, num_datapoints=20
):
    num_samples = 5
    ground_truth_num_samples = 100
    algorithms = [
        # "NN",
        f"IS",
        # f"IS+NN",
        # f"SMC",
        # f"SMC+NN",
        # f"SMC+NN+rejuvenation",
        # f"Online_IS+NN",
    ]
    ground_truth_algorithm = f"SMC(100)"
    probss = {}
    l2 = {}

    # for i, (states, actions, num_block_pairs, desires) in enumerate(dataset):
    for i in range(1):
        (states, actions, num_block_pairs, desires) = dataset[i]
        if i >= num_datapoints:
            break
        print(f"DATA POINT {i}")

        desire_int = desires[0].item()
        # -- Extract values
        num_possible_block_pairs = num_block_pairs.item()
        assert all(element == desires[0] for element in desires)
        desire_util = utils.construction_data.desire_int_to_utilities(desire_int, num_possible_block_pairs)

        # Compute posteriors using different algorithms
        for algorithm in algorithms + [ground_truth_algorithm]:
            probs = get_particle_inference_probs(
                states, actions, desires, algorithm, num_samples=num_samples, model=model
            )
            if not (algorithm in probss):
                probss[algorithm] = {}
            # [num_timesteps, num_possible_rankings = num_possible_food_trucks!]
            probss[algorithm][i] = probs
        # Computer posterior errors
        for algorithm in algorithms:
            if not (algorithm in l2):
                l2[algorithm] = {}
            # [num_timesteps]
            l2[algorithm][i] = get_l2(probss[ground_truth_algorithm][i], probss[algorithm][i])

        # Plot posterior error distribution as we go, as long as we have 2
        print(f"PLOTTING POSTERIOR ERROR at {posterior_error_path}")
        if i >= 1:
            y0s = np.arange(len(algorithms))
            height = 0.8

            fig, ax = plt.subplots(1, 1)
            for algorithm, y0 in zip(algorithms, y0s):
                data = [np.mean(v) for k, v in l2[algorithm].items()]
                data = [x for x in data if not np.isinf(x)]
                data = np.array(data)
                if len(data) > 0:
                    # Plot the histogram and individual errors
                    plot_hist(data, y0, height, ax)
                    ax.scatter(
                        data,
                        [y0 + 0.1 for _ in range(len(data))],
                        s=10.0,
                        color="black",
                        zorder=100,
                    )
                    # Annotate each individual error
                    for x_idx, x in l2[algorithm].items():
                        ax.text(
                            np.mean(x),
                            y0 + 0.2,
                            f"{x_idx}",
                            horizontalalignment="center",
                            verticalalignment="bottom",
                            fontsize=3,
                        )

                # Draw a mean line
                ax.plot([np.mean(data) for _ in range(2)], [y0, y0 + 1], color="C0")

                # Separate different algorithms
                ax.axhline(y0, color="C0")

            # Format figure
            ax.set_yticks(y0s)
            ax.set_yticklabels(algorithms)
            ax.set_ylim(0, y0s[-1] + 1)
            ax.tick_params(length=0, axis="y")
            ax.set_xlabel(
                f"L2 between the desire posteriors (averaged over time)\n"
                f"from different algorithms and ground truth ({ground_truth_algorithm})"
            )
            ax.set_ylabel("Algorithm")
            ax.set_title("Distribution of posterior errors")
            utils.general.save_fig(
                fig, posterior_error_path,
            )

        # Plot interaction gif
        # -- What's the gif path
        posterior_gif_path = f"{posterior_gifs_dir}/{i}.gif"
        print(f"PLOTTING INTERACTION GIF to {posterior_gif_path}")

        tmp_dir = utils.general.get_tmp_dir()
        # -- Precompute xticklabels
        utilities_permutations = list(
            itertools.permutations(envs.construction.ALL_UTILITIES[:num_possible_block_pairs])
        )
        final_util_perms = []
        for util in utilities_permutations:
            if util not in final_util_perms:
                final_util_perms.append(util)

        x_tick_labels = []
        for utilities_permutation in final_util_perms:
            name_util_mapping = []
            for colored_block, utility in zip(envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs],
                                              utilities_permutation):
                if utility != 0:
                    block1, block2 = block2color[colored_block[0]], block2color[colored_block[1]]
                    block_pair = (block1, block2)
                    name_util_mapping.append(f"{block_pair}={utility}")
            x_tick_labels.append(", ".join(name_util_mapping))
        # pdb.set_trace()

        # -- Plot
        img_paths = []
        for timestep in range(len(states)):
            state = states[timestep]
            action = actions[timestep]

            fig, axs = plt.subplots(1, len(algorithms) + 2, figsize=(5 * (len(algorithms) + 2), 6))

            # Current state and action
            ax = axs[0]
            raw_state = utils.construction_data.state_tensor_to_state(state)
            raw_state.plot(ax=ax)
            block_picked = raw_state.block_picked
            if block_picked is not None:
                block_picked = block2color[block_picked]
            ax.set_title(f"\nNext action: {utils.construction_data.action_tensor_to_action(action).name}"
                         f"\nAgent Inventory: {block_picked}")

            # Color for bar plots
            color = [
                "C0" for _ in range(envs.construction.get_num_rankings(num_possible_block_pairs))
            ]

            color[desire_int] = "black"

            # Ground truth string
            ground_truth_pair = []
            for k, v in utils.construction_data.desire_int_to_utilities(desire_int, num_possible_block_pairs).items():
                if v != 0:
                    block1, block2 = block2color[k[0]], block2color[k[1]]
                    block_pair = (block1, block2)
                    ground_truth_pair.append(f"{block_pair}={v}")
            ground_truth_str = ", ".join(ground_truth_pair)
            # pdb.set_trace()
            # Plot posteriors from different algorithms
            for ax, algorithm in zip(axs[1:], algorithms + [ground_truth_algorithm]):
                probs = probss[algorithm][i][timestep]
                ax.bar(np.arange(len(probs)), probs, color=color)
                if algorithm != ground_truth_algorithm:
                    ax.set_title(
                        f"{algorithm} (Error = {np.mean(l2[algorithm][i]):.2f})\n"
                        f"Ground truth: {ground_truth_str}"
                    )
                else:
                    ax.set_title(f"{algorithm}\nGround truth: {ground_truth_str}")

            # Formatting axes
            for ax in axs:
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 1])
                ax.tick_params(length=0)
                ax.tick_params(axis="x", labelrotation=90)
                ax.set_xticks(range(len(probs)))
                ax.set_xticklabels(x_tick_labels)

            # Save to png
            img_path = f"{tmp_dir}/{timestep}.png"
            utils.general.save_fig(fig, img_path)
            img_paths.append(img_path)

        # Make gif
        utils.general.make_gif(img_paths, posterior_gif_path, 3)
        shutil.rmtree(tmp_dir)

def plot_reasoning_about_L0_top(
    posterior_error_path, posterior_gifs_dir, dataset, model, num_datapoints=20
):
    algorithms = [
        "NN",
    ]

    probss = {}

    for i, (states, actions, desires) in enumerate(dataset):
        if i >= num_datapoints:
            break

        print(f"DATA POINT {i}")
        # Compute posteriors using different algorithms
        for algorithm, probs in zip(algorithms, [get_nn_probs(model, states, actions),],):
            if not (algorithm in probss):
                probss[algorithm] = {}
            # [num_timesteps, num_possible_rankings = num_possible_block_pairs!]
            probss[algorithm][i] = probs

        # Plot interaction gif
        # -- What's the gif path
        posterior_gif_path = f"{posterior_gifs_dir}/{i}.gif"
        print(f"PLOTTING INTERACTION GIF to {posterior_gif_path}")

        # -- Extract values
        num_possible_colored_blocks = int((states.shape[-1] - 4) / 2)
        num_possible_block_pairs = int(num_possible_colored_blocks * (num_possible_colored_blocks - 1) / 2)
        desire_int = desires[0].item()
        tmp_dir = utils.general.get_tmp_dir()

        # -- Precompute xticklabels
        x_tick_labels = envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs]
        # -- Plot

        img_paths = []
        for timestep in range(len(states)):
            state = states[timestep]
            action = actions[timestep]

            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            # Current state and action
            ax = axs[0]
            utils.construction_data.state_tensor_to_state(state).plot(ax=ax)
            ax.set_title(f"\nNext action:\n{utils.construction_data.action_tensor_to_action(action).name}")

            # Color for bar plots
            color = [
                "C0" for _ in range(num_possible_block_pairs)
            ]
            color[desire_int] = "black"

            # Ground truth string
            ground_truth_str = envs.construction.ALL_BLOCK_PAIRS[desire_int]

            # Plot posteriors from different algorithms
            for ax, algorithm in zip(axs[1:], algorithms):
                probs = probss[algorithm][i][timestep]
                ax.bar(np.arange(len(probs)), probs, color=color)
                ax.set_title(f"{algorithm}\n" f"Ground truth: {ground_truth_str}")

            # Formatting axes
            for ax in axs:
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 1])
                ax.tick_params(length=0)
                ax.tick_params(axis="x", labelrotation=90)
                ax.set_xticks(range(len(probs)))
                ax.set_xticklabels(x_tick_labels)

            # Save to png
            img_path = f"{tmp_dir}/{timestep}.png"
            utils.general.save_fig(fig, img_path)
            img_paths.append(img_path)

        # Make gif
        utils.general.make_gif(img_paths, posterior_gif_path, 3)
        shutil.rmtree(tmp_dir)

def plot_stats(path, stats):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    ax = axs[0]
    ax.plot(stats["losses"])
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")

    ax = axs[1]
    ax.plot(stats["train_accuracies"], label="Train")
    ax.plot(stats["test_accuracies"], label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()

    utils.general.save_fig(
        fig, path,
    )

def get_checkpoint_paths(save_dir, env_name, experiment_name):
    experiment_dir = f"{save_dir}/{env_name}/{experiment_name}/"
    if Path(experiment_dir).exists():
        for config_name in sorted(os.listdir(experiment_dir)):
            # yield f"{experiment_dir}/{config_name}/checkpoints/latest.pik"
            yield f"{experiment_dir}/{config_name}/checkpoints/best_acc.pik"
    else:
        return []

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
            get_checkpoint_paths(args.save_dir, args.env_name, args.experiment_name)
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

        # Load dataset
        dataset_dir = f"data/{train_args.env_name}"

        dataset_train = utils.construction_data.ReasoningAboutL0Dataset(
            num_colored_blocks=train_args.num_colored_block_locations,
            num_possible_block_pairs=train_args.num_possible_block_pairs,
            num_rows=train_args.num_rows,
            num_cols=train_args.num_cols,
            beta=train_args.beta,
            utility_mode=train_args.utility_mode,
            num_data=train_args.num_data_train,
            dataset_dir=dataset_dir,
            train=True,
            device=device,
        )
        dataset_train.load()

        dataset_test = utils.construction_data.ReasoningAboutL0Dataset(
            num_colored_blocks=train_args.num_colored_block_locations,
            num_possible_block_pairs=train_args.num_possible_block_pairs,
            num_rows=train_args.num_rows,
            num_cols=train_args.num_cols,
            beta=train_args.beta,
            utility_mode=train_args.utility_mode,
            num_data=train_args.num_data_test,
            dataset_dir=dataset_dir,
            train=False,
            device=device,
        )
        dataset_test.load()

        # Plot posterior error distribution and posterior gifs
        for train_or_test_str, dataset in zip(["test", "train"], [dataset_test, dataset_train]):
            if train_args.utility_mode == "ranking":
                plot_reasoning_about_L0(
                    f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                    f"{config_name}/reasoning_about_L0/posterior_error_distribution/"
                    f"{train_or_test_str}.pdf",
                    f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                    f"{config_name}/reasoning_about_L0/posterior_gifs/{train_or_test_str}",
                    dataset,
                    model,
                    num_datapoints=20,
                )
            if train_args.utility_mode == "top":
                plot_reasoning_about_L0_top(
                    f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                    f"{config_name}/reasoning_about_L0/posterior_error_distribution/"
                    f"{train_or_test_str}.pdf",
                    f"{train_args.save_dir}/{train_args.env_name}/{train_args.experiment_name}/"
                    f"{config_name}/reasoning_about_L0/posterior_gifs/{train_or_test_str}",
                    dataset,
                    model,
                    num_datapoints=20,
                )
        plotted_something = True
    return plotted_something

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=123, help="Random seed")
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
