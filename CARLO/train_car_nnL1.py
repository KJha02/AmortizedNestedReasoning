from car_utils.car_data import car_collate, joint_sa_tensor_to_state_action
from car_utils.car_data import ReasoningAboutScenario2L1Dataset, ReasoningAboutScenario1L0Dataset
from tqdm import tqdm
import torch
import argparse
import pdb
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from car_models.ToMnet_car import ToMnet_car_pred
import car_utils.network as network
import car_utils.general
import time
import random
import numpy as np
from test_scenario1 import percentCorrect
from train_belief_nn import load_belief_checkpoint

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--env-name", type=str, default="scenario2", help="Environment name")
    parser.add_argument("--save-dir", type=str, default="save", help="Save directory")
    parser.add_argument("--experiment-name", type=str, default="", help="Experiment name")
    parser.add_argument("--action-size", type=int, default=5, help="Action space size")
    parser.add_argument("--goal-size", type=int, default=3, help="Goal space size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-data-train", type=int, default=1, help="Training set size")
    parser.add_argument("--num-data-test", type=int, default=1, help="Testing set size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.01, help="How deterministic is the policy")
    parser.add_argument("--num-samples", type=int, default=1, help="How many particles are sampled in L0 IS_inference")
    parser.add_argument("--num-samples-L2", type=int, default=3, help="How many particles are sampled in L1 IS_inference")
    parser.add_argument("--sampled-actions", type=int, default=1, help="How actions to sample in policy")
    parser.add_argument("--lookAheadDepth", type=int, default=10, help="How far ahead to look when planning")
    parser.add_argument('--predictActions', action='store_true')
    parser.set_defaults(predictActions=False)

    parser.add_argument('--kunalDir', action='store_true')
    parser.set_defaults(kunalDir=False)

    parser.add_argument('--cross-entropy', action='store_true')
    parser.set_defaults(cross_entropy=False)
    
    return parser

def get_config_name(args):
    return (
        f"num_sampled_actions={args.sampled_actions},"
        f"lookAheadDepth={args.lookAheadDepth},"
        f"beta={args.beta}"
    )

def init_actionPred(args, device='cpu', output_dim=None):
    if output_dim is None:
        action_model = ToMnet_car_pred(hidden_dim=args.hidden_dim, output_dim=args.goal_size)
    else:
        action_model = ToMnet_car_pred(hidden_dim=args.hidden_dim, output_dim=output_dim)
    action_model.to(device)
    action_optimizer = optim.Adam(action_model.parameters(), lr=args.lr)
    return action_model, action_optimizer

def save_checkpoint(path, model, optimizer, stats, args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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

def load_checkpoint(path, device, num_tries=3, L1=False, actionPred=False):
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
    if actionPred:
        model, optimizer = init_actionPred(args, device, output_dim=5)
    else:
        model, optimizer = init_actionPred(args, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, args

def main(args):
    if args.experiment_name == "":
        raise RuntimeError("Provide --experiment-name")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        device = "cuda"
    else:
        device = "cpu"

    if args.kunalDir:
        BIG_STORAGE_DIR = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning/"
    else:
        BIG_STORAGE_DIR = "../"

    L0_model_save_path = f"{BIG_STORAGE_DIR}CARLO/save/scenario1/debug/num_sampled_actions=1,lookAheadDepth=1,beta=0.01/checkpoints/best_acc.pik"
    L0_inference_model, L0_optimizer, L0_stats, L0_args = load_checkpoint(L0_model_save_path, device)

    state_belief_save_dir = f"{BIG_STORAGE_DIR}CARLO/save/stateEstimation/debug/num_sampled_actions=1,lookAheadDepth=1,beta=0.01/checkpoints/"
    state_model_path = state_belief_save_dir + "best_acc_state.pik"
    exist_model_path = state_belief_save_dir + "best_acc_exist.pik"
    state_model, _, _, _ = load_belief_checkpoint(state_model_path, device, exist_model=False)
    exist_model, _, _, _ = load_belief_checkpoint(exist_model_path, device, exist_model=True)


    L1_model_save_dir = f"{BIG_STORAGE_DIR}CARLO/{args.save_dir}/{args.env_name}/{args.experiment_name}/{get_config_name(args)}"
    p = Path(L1_model_save_dir)
    if not p.is_dir():
        p.mkdir(parents=True)
    dataset_dir = f"{BIG_STORAGE_DIR}data/{args.env_name}"

    train_dataset = ReasoningAboutScenario2L1Dataset(
        beta=args.beta,
        num_data=args.num_data_train,
        seed=args.seed,
        dataset_dir=dataset_dir,
        train=True,
        device=device,
        num_inference_samples=args.num_samples_L2,
        sampled_actions = args.sampled_actions,
        lookAheadDepth = args.lookAheadDepth,
        car1_exist_prior=0.65,
        car2_exist_prior=0.65,
        L0_inference_model=L0_inference_model,
        other_agent_inference_algorithm="Online_IS+NN",
        other_agent_num_samples=args.num_samples,
        state_model=state_model,
        exist_model=exist_model
    )
    train_dataset.load()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=car_collate,
        shuffle=False,
    )
    test_dataset = ReasoningAboutScenario2L1Dataset(
        beta=args.beta,
        num_data=args.num_data_train,
        seed=args.seed,
        dataset_dir=dataset_dir,
        train=False,
        device=device,
        num_inference_samples=args.num_samples_L2,
        sampled_actions = args.sampled_actions,
        lookAheadDepth = args.lookAheadDepth,
        car1_exist_prior=0.65,
        car2_exist_prior=0.65,
        L0_inference_model=L0_inference_model,
        other_agent_inference_algorithm="Online_IS+NN",
        other_agent_num_samples=args.num_samples,
        state_model=state_model,
        exist_model=exist_model
    )
    test_dataset.load()

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=car_collate,
        shuffle=False,
    )


    stats = {"train_losses": [], "test_losses": [], "train_accuracies":[], "test_accuracies":[]}

    if args.predictActions:
        model, optimizer = init_actionPred(args, device, output_dim=args.action_size)
    else:
        model, optimizer = init_actionPred(args, device)
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    NLL_loss = torch.nn.NLLLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

    min_test_loss = float("inf")
    max_test_acc = float("-inf")
    for epoch_id in tqdm(range(args.num_epochs)):
        # training run
        temp_L = 0
        numBatches = 0
        model.train()
        avgAccuracy = 0.0
        data_considered = 0
        for batch_id, batch in enumerate(train_dataloader):
            if data_considered >= (100 * len(train_dataset) // 100):  # training on less data
                data_considered = 0
                break
            else:
                data_considered += len(batch)

            state_actions, id_pair, IS_goal_inferences, IS_action_inferences, other_agent_goal, other_agent_actions = batch
            lens = torch.LongTensor([s.shape[0] for s in state_actions]).cpu()
            with torch.cuda.amp.autocast():
                log_prob = model(state_actions, id_pair, lens)
                if args.cross_entropy:
                    if args.predictActions:
                        loss = NLL_loss(log_prob, torch.tensor(other_agent_actions, device=device))
                    else:
                        loss = NLL_loss(log_prob, torch.tensor(other_agent_goal, device=device))
                else:
                    if args.predictActions:
                        target = torch.stack(IS_action_inferences, dim=0)
                    else:
                        target = torch.stack(IS_goal_inferences, dim=0)
                    loss = kl_loss(log_prob, target)
                

            avgAccuracy += percentCorrect(log_prob, other_agent_goal)

            network.update_network(loss.double(), optimizer, scaler=scaler, model=model)

            temp_L += loss.item()
        numBatches += len(train_dataloader)

        avgAccuracy /= numBatches
        avgLoss = temp_L / numBatches
        stats["train_losses"].append(avgLoss)
        stats["train_accuracies"].append(avgAccuracy)

        # testing run
        temp_L = 0
        numBatches = 0
        model.eval()
        avgAccuracy = 0.0
        for batch_id, batch in enumerate(test_dataloader):
            state_actions, id_pair, IS_goal_inferences, IS_action_inferences, other_agent_goal, other_agent_actions = batch
            lens = torch.LongTensor([s.shape[0] for s in state_actions]).cpu()
            with torch.cuda.amp.autocast():
                log_prob = model(state_actions, id_pair, lens)
                if args.cross_entropy:
                    if args.predictActions:
                        loss = NLL_loss(log_prob, torch.tensor(other_agent_actions, device=device))
                    else:
                        loss = NLL_loss(log_prob, torch.tensor(other_agent_goal, device=device))
                else:
                    if args.predictActions:
                        target = torch.stack(IS_action_inferences, dim=0)
                    else:
                        target = torch.stack(IS_goal_inferences, dim=0)
                    loss = kl_loss(log_prob, target)
                
            avgAccuracy += percentCorrect(log_prob, other_agent_goal)

            temp_L += loss.item()
            numBatches += 1

        avgAccuracy /= numBatches
        avgLoss = temp_L / numBatches
        stats["test_losses"].append(avgLoss)
        stats["test_accuracies"].append(avgAccuracy)

        if avgAccuracy > max_test_acc:
            max_test_acc = avgAccuracy
            save_checkpoint(f"{L1_model_save_dir}/checkpoints/best_acc.pik", model, optimizer, stats, args)

        if epoch_id % 50 == 0:
            save_checkpoint(f"{L1_model_save_dir}/checkpoints/epoch_{epoch_id}.pik", model, optimizer, stats, args)


        if avgLoss < min_test_loss:
            min_test_loss = avgLoss
            save_checkpoint(f"{L1_model_save_dir}/checkpoints/best_loss.pik", model, optimizer, stats, args)

        if epoch_id % 10 == 0:
            print(f"Avg Train Loss Epoch {epoch_id}: {stats['train_losses'][-1]} ; Avg Test Loss Epoch {epoch_id}: {stats['test_losses'][-1]}")

            # Plot stats
            fig, axss = plt.subplots(
                1, 2, figsize=(2 * 8, 2 * 6), squeeze=False
            )
            axs = axss.flatten()
            # assert len(stats["train_accuracies"]) == epoch_id + 1
            ax = axs[0]
            ax.plot(stats["train_losses"], label="Train")
            ax.plot(stats["test_losses"], label="Test")
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('NNL3 NLL Loss')
            ax.grid(True)

            ax = axs[1]
            ax.plot(stats["train_accuracies"], label="Train")
            ax.plot(stats["test_accuracies"], label="Test")
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('NNL3 Action Prediction Accuracy')
            ax.grid(True)

            car_utils.general.save_fig(fig, f"{L1_model_save_dir}/stats.png")
            



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)