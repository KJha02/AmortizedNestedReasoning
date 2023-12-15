import matplotlib
matplotlib.use('Agg')
import random
import numpy as np 
from scipy.special import rel_entr
import torch 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train_belief_nn import load_belief_checkpoint
from train_car_nnL1 import load_checkpoint
from test_scenario1 import percentCorrect
from car_utils.car_data import car_collate, joint_sa_tensor_to_state_action, joint_pair_tensor_to_IDs
from car_utils.car_data import ReasoningAboutScenario2L1Dataset
from reasoning_about_car_L1 import get_L1_online_inference, get_car_L1_is_inference
from scenario import Action
import car_utils.general as general
import pdb
import shutil
import _pickle as pickle
from agents import Car
from tqdm import tqdm

def make_car_gif(gif_path, save_dir, states_raw):
	from visualizer import Visualizer
	tmp_dir = general.get_tmp_dir()
	img_paths  = []
	for timestep, state in enumerate(states_raw):
		state.visualizer = Visualizer(state.width, state.height, ppm=state.ppm)
		state.render(save_dir + tmp_dir, timestep)
		img_paths.append(f"{save_dir}{tmp_dir}/{timestep}_gui.png")

	general.make_gif(img_paths, save_dir + gif_path, 3)
	shutil.rmtree(save_dir + tmp_dir)

def eliminate_zero(distribution):
	'''
	Adds small noise to eliminate zeroed out values and make kl divergence stable
	'''
	num_nonzeros = np.count_nonzero(distribution)
	if num_nonzeros == 0:  # everything is 0, so make uniform
		dist_length = len(distribution)
		distribution = [1/dist_length] * dist_length
	else:
		num_zeros = len(distribution) - num_nonzeros # find number of zeros 
		if num_zeros > 0:
			noise_subtraction = (1e-6 * num_zeros) / num_nonzeros  # take away uniformly from non-zeros so we sum to 1
			for j, prob in enumerate(distribution):
				if prob == 0:  # add some noise
					distribution[j] = 1e-6
				else:
					distribution[j] -= noise_subtraction

def KL(gt_distrib, approx_distrib):
	'''
	Calculates KL divergence between 1d arrays
	'''
	eliminate_zero(gt_distrib)
	eliminate_zero(approx_distrib)
	return sum(rel_entr(gt_distrib, approx_distrib))

def avg_KL(gt_distribs, approx_distribs):
	'''
	Calculates average KL divergence between 2d arrays
	'''
	return np.mean([KL(gt_distrib, approx_distribs[i]) for i, gt_distrib in enumerate(gt_distribs)])

def numberCorrect(predictions, target):
	'''
	Returns average number of instances in which max probability proposal was true utility
	'''
	if type(target) == list and type(target) != torch.Tensor:
		correct_classes = np.array(target)
	elif type(target) == torch.Tensor:
		correct_classes = target.cpu().detach().numpy()
	else:
		correct_classes = np.array([target] * len(predictions))

	if type(predictions) == list:
		predictions = np.array(predictions)
	if type(predictions) == torch.Tensor:
		predictions = predictions.cpu().detach().numpy()

	num_samples, num_classes = predictions.shape
	correct_count = 0

	for i in range(num_samples):
		ground_truth = correct_classes[i]
		max_probabilities = [prob for prob in predictions[i] if prob == max(predictions[i])]
		num_max_probs = len(max_probabilities)

		# Handle tie-breakers
		if ground_truth in [idx for idx, prob in enumerate(predictions[i]) if prob == max(predictions[i])]:
			correct_count += 1 / num_max_probs

	accuracy = correct_count / num_samples
	return accuracy


# setting seed
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
cuda = torch.cuda.is_available()
if cuda:
	torch.cuda.manual_seed(123)
	device = "cuda"
else:
	device = "cpu"


# load models
BIG_STORAGE_DIR = "../"
# BIG_STORAGE_DIR = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning/"

L0_model_save_path = f"{BIG_STORAGE_DIR}CARLO/save/scenario1/debug/num_sampled_actions=1,lookAheadDepth=1,beta=0.01/checkpoints/best_acc.pik"
L0_inference_model, L0_optimizer, L0_stats, L0_args = load_checkpoint(L0_model_save_path, device)
L0_inference_model.eval()

state_belief_save_dir = f"{BIG_STORAGE_DIR}CARLO/save/stateEstimation/debug/num_sampled_actions=1,lookAheadDepth=1,beta=0.01/checkpoints/"
state_model_path = state_belief_save_dir + "best_acc_state.pik"
exist_model_path = state_belief_save_dir + "best_acc_exist.pik"
state_model, _, _, _ = load_belief_checkpoint(state_model_path, device, exist_model=False)
exist_model, _, _, _ = load_belief_checkpoint(exist_model_path, device, exist_model=True)
state_model.eval()
exist_model.eval()

L1_model_save_dir = f"{BIG_STORAGE_DIR}CARLO/save/scenario2/debug/num_sampled_actions=1,lookAheadDepth=10,beta=0.01"
L1_model_save_path = f"{L1_model_save_dir}/checkpoints/best_acc.pik"
L1_inference_model, L1_optimizer, L1_stats, L1_args = load_checkpoint(L1_model_save_path, device)
L1_inference_model.eval()

L1_nll_model_dir = f"{BIG_STORAGE_DIR}CARLO/save/scenario2/debug_nll_goals/num_sampled_actions=1,lookAheadDepth=10,beta=0.01"
L1_nll_model_path = f"{L1_model_save_dir}/checkpoints/best_acc.pik"
cross_ent_model, cross_ent_optimizer, cross_ent_stats, cross_ent_args = load_checkpoint(L1_nll_model_path, device, L1=True)
cross_ent_model.eval()

L1_nll_model_action_dir  = f"{BIG_STORAGE_DIR}CARLO/save/scenario2/debug_nll_actions/num_sampled_actions=1,lookAheadDepth=10,beta=0.01"
L1_nll_model_action_path  = f"{L1_nll_model_action_dir}/checkpoints/best_acc.pik"
cross_ent_action_model, _, _, _ = load_checkpoint(L1_nll_model_action_path, device, L1=True, actionPred=True)
cross_ent_action_model.eval()

# load dataset
dataset_dir = f"{BIG_STORAGE_DIR}data/{L1_args.env_name}"

test_dataset = ReasoningAboutScenario2L1Dataset(
	beta=L1_args.beta,
	num_data=1,
	seed=L1_args.seed,
	dataset_dir=dataset_dir,
	train=False,
	device=device,
	num_inference_samples=L1_args.num_samples_L2,
	sampled_actions = L1_args.sampled_actions,
	lookAheadDepth = L1_args.lookAheadDepth,
	car1_exist_prior=0.65,
	car2_exist_prior=0.65,
	L0_inference_model=L0_inference_model,
	other_agent_inference_algorithm="Online_IS+NN",
	other_agent_num_samples=L1_args.num_samples,
	state_model=state_model,
	exist_model=exist_model
)
beta_L0 = 0.001
beta_L1 = 0.001
# # this dataset is for nearsighted rando drivers
# test_dataset = ReasoningAboutScenario2L1Dataset(
# 	beta=0.01,
# 	num_data=21,
# 	seed=L1_args.seed,
# 	dataset_dir=dataset_dir,
# 	train=False,
# 	device=device,
# 	num_inference_samples=1,
# 	sampled_actions = L1_args.sampled_actions,
# 	lookAheadDepth = 1,
# 	car1_exist_prior=0.0,
# 	car2_exist_prior=0.0,
# 	L0_inference_model=L0_inference_model,
# 	other_agent_inference_algorithm="Online_IS+NN",
# 	other_agent_num_samples=1,
# 	state_model=state_model,
# 	exist_model=exist_model
# )
# beta_L0 = 0.01
# beta_L1 = 0.01

test_dataset.load()

test_dataloader = DataLoader(
	test_dataset,
	batch_size=1,
	collate_fn=car_collate,
	shuffle=False,
)


# initialize accuracy metrics
nn_accuracy = []
nn_kl_div = []
full_is_accuracy = []


num_predictions = 0
num_batches = 0


accuracyDictList = {}
klDict = {}
avgAccuracyPctTime = {}
avgAccuracyPctTime['is'] = {i: [] for i in range(10)}
avgAccuracyPctTime['nn'] = {i: [] for i in range(10)}
for L2_sample in range(1, 4):
	for L1_sample in range(1, 4):
		accuracyDictList[(L2_sample, L1_sample)] = []
		accuracyDictList[('random', L2_sample*L1_sample)] = []
		klDict[(L2_sample, L1_sample)] = []
		klDict[('random', L2_sample*L1_sample)] = []
		avgAccuracyPctTime[(L2_sample, L1_sample)] = {i: [] for i in range(10)}
accuracyDictList["is"] = []
accuracyDictList["nn"] = []

sampled_batch = random.sample(range(0, 99), 10)
action_space = list(Action)
for batch_id, batch in enumerate(test_dataloader):
	if batch_id == 100:
		break

	rawInf = {}


	state_actions, id_pair, IS_goal_inferences, IS_action_inferences, other_agent_goal, other_agent_actions = batch
	lens = torch.LongTensor([s.shape[0] for s in state_actions]).cpu()

	# NLL Loss Baseline
	cross_ent_log_prob = cross_ent_action_model(state_actions, id_pair, lens)
	cross_ent_nn_inferences = torch.softmax(cross_ent_log_prob, 1).cpu().detach().numpy()
	cross_ent_nn_inferences /= cross_ent_nn_inferences.sum(axis=1, keepdims=True)
	rawInf["nn"] = cross_ent_nn_inferences

	# setting up tensors for inference
	inference_pair = joint_pair_tensor_to_IDs(id_pair[0])
	baseID = inference_pair[0]
	targetAgentID = inference_pair[1]
	goal_int = other_agent_goal[0].item()
	state_tensor_rollouts = state_actions[0]
	states_raw = [joint_sa_tensor_to_state_action(s) for s in state_tensor_rollouts]
	other_agent_actions_raw = [action_space[a] for a in other_agent_actions]
	init_state, init_actions, env = joint_sa_tensor_to_state_action(state_tensor_rollouts[0], return_scenario=True)

	# Exact Inference Baseline - consider all 72 particles
	try:
		full_IS_goal_inferences, full_IS_action_inferences = get_car_L1_is_inference(env, 
			states_raw, other_agent_actions_raw, targetAgentID, carExistPrior=0.65, 
			num_samples=3, sampled_actions=10, lookAheadDepth=10, 
			other_agent_num_samples=3, beta_L0=beta_L0, beta_L1=beta_L1,
			other_agent_inference_algorithm="IS", L0_inference_model=None, signal_danger_prior=0.5,
			state_model=None, exist_model=None)
		#rawInf["is"] = full_IS_goal_inferences
		rawInf["is"] = full_IS_action_inferences
	except:
		continue

	temp_accuracy = numberCorrect(cross_ent_nn_inferences, other_agent_actions)
	temp_kl = avg_KL(full_IS_action_inferences, cross_ent_nn_inferences)
	nn_kl_div.append(temp_kl)
	accuracyDictList["nn"].append(temp_accuracy)

	# temp_accuracy = numberCorrect(full_IS_goal_inferences, goal_int)
	temp_accuracy = numberCorrect(full_IS_action_inferences, other_agent_actions)
	accuracyDictList["is"].append(temp_accuracy)


	# Online IS + NN
	log_prob = L1_inference_model(state_actions, id_pair, lens)
	nn_inferences = torch.softmax(log_prob, 1).cpu().detach().numpy()
	nn_inferences /= nn_inferences.sum(axis=1, keepdims=True)

	failed = False
	# for L2_sample in range(2, 3):
	# 	for L1_sample in range(1, 2):
	for L2_sample in tqdm(range(1, 4)):
		if failed:
			break
		for L1_sample in tqdm(range(1, 4)):
			try:
				goal_inf, action_inf = get_L1_online_inference(env, 
					states_raw, other_agent_actions_raw, targetAgentID, carExistPrior=0.65, 
					num_samples=L2_sample, sampled_actions=10, lookAheadDepth=10, 
					other_agent_num_samples=L1_sample, beta_L0=beta_L0, beta_L1=beta_L1, 
					other_agent_inference_algorithm="Online_IS+NN", 
					L0_inference_model=L0_inference_model, signal_danger_prior=0.5, 
					state_model=state_model, exist_model=exist_model, 
					lane_utilities_proposal_probss=nn_inferences)
			except:
				failed = True 
				break
			# rawInf[(L2_sample, L1_sample)] = goal_inf
			rawInf[(L2_sample, L1_sample)] = action_inf
			
			# temp_accuracy = numberCorrect(goal_inf, goal_int)
			# temp_kl = avg_KL(full_IS_goal_inferences, goal_inf)
			temp_accuracy = numberCorrect(action_inf, other_agent_actions)
			temp_kl = avg_KL(full_IS_action_inferences, action_inf)
			accuracyDictList[(L2_sample, L1_sample)].append(temp_accuracy)
			klDict[(L2_sample, L1_sample)].append(temp_kl)

			try:
				random_goal_inf, random_action_inf = get_L1_online_inference(env, 
					states_raw, other_agent_actions_raw, targetAgentID, carExistPrior=0.65, 
					num_samples=L2_sample, sampled_actions=10, lookAheadDepth=10, 
					other_agent_num_samples=L1_sample, beta_L0=beta_L0, beta_L1=beta_L1, 
					other_agent_inference_algorithm="random", 
					L0_inference_model=L0_inference_model, signal_danger_prior=0.5, 
					state_model=state_model, exist_model=exist_model, 
					lane_utilities_proposal_probss=nn_inferences,
					randomChoice=True)
			except:
				break
			rawInf[('random', L2_sample*L1_sample)] = random_action_inf

			# temp_accuracy = numberCorrect(goal_inf, goal_int)
			# temp_kl = avg_KL(full_IS_goal_inferences, goal_inf)
			temp_accuracy = numberCorrect(action_inf, other_agent_actions)
			temp_kl = avg_KL(full_IS_action_inferences, action_inf)
			accuracyDictList[('random', L2_sample * L1_sample)].append(temp_accuracy)
			klDict[('random', L2_sample*L1_sample)].append(temp_kl)

	if failed:
		continue

	# Storing accuracy for all benchmarks
	for key, inf in rawInf.items():
		if key[0] == 'random':
			continue
		for t in range(1, len(inf) + 1):
			for pct in range(1, 11):
				if (t <= pct * len(inf) / 10):
					# avgAccuracyPctTime[key][pct-1].append(numberCorrect(inf[t-1:t], goal_int))
					avgAccuracyPctTime[key][pct-1].append(numberCorrect(inf[t-1:t], other_agent_actions[t-1:t]))
					break

	# plotting
	continue
	if (batch_id in sampled_batch):
		path = f'save/examples/specificAccuracyRando{batch_id}.png'

		actualStates = []
		for state in states_raw:
			s = state[0].clone()
			s.dynamic_agents = []
			x = 'blue'
			for t in state[0].dynamic_agents:
				if t.ID == baseID:
					x = 'green'
				elif t.ID == targetAgentID:
					x = 'red'
				else:
					x = 'blue'
	 
				s.add(Car(t.center, t.heading, ID=t.ID, color=x))

			actualStates.append(s)
		make_car_gif(f"specificAccuracyRando{batch_id}.gif", "save/examples/", actualStates)
		# want to generate plot of accuracy metrics vs time
		num_cols = 1
		num_rows = 1
		fig, axss = plt.subplots(
			num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), squeeze=False
		)
		axs = axss.flatten()

		key = (2, 3)
		inf = rawInf[key]
		timestepLabels = list(range(len(inf)))

		forwardProbs = [x[0] for x in inf]
		leftProbs = [x[1] for x in inf]
		rightProbs = [x[2] for x in inf]
		stopProbs = [x[3] for x in inf]
		signalProbs = [x[4] for x in inf]

		model_label = str(key[0] * key[1]) + " particle(s)"
		axs[0].plot(timestepLabels, forwardProbs, label='Forward')
		axs[0].plot(timestepLabels, leftProbs, label='Left')
		axs[0].plot(timestepLabels, rightProbs, label='Right')
		axs[0].plot(timestepLabels, stopProbs, label='Stop')
		axs[0].plot(timestepLabels, signalProbs, label='Signal')

		potential_labels = ["Forward", "Left", "Right", "Stop", "Signal"]
		#goal_label = potential_labels[goal_int]

		#axs[0].set_title(f"Inferred Ground Truth ({goal_label}) Probability by Time for {model_label}")
		axs[0].set_title(f"Inferred next action probabilities by Time")
		axs[0].set_ylabel("Probability")
		axs[0].set_xlabel("Timestep")
		axs[0].legend()

		general.save_fig(fig, path, tight_layout_kwargs={"rect": [0, 0, 1, 1]})

print("Final plotting")

num_cols = 1
num_rows = 1
fig, axss = plt.subplots(
	num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), squeeze=False
)
axs = axss.flatten()
x_ticks = [f'{x}%' for x in range(10, 110, 10)]
for key, timeDict in avgAccuracyPctTime.items():
	xvals = []
	yvals = []
	yerrs = []
	for x, accs in timeDict.items():
		if len(accs) == 0:
			continue
		xvals.append(x)
		yvals.append(np.mean(accs))
		yerrs.append(np.std(accs))

	if key == 'nn':
		label = 'NN'
		axs[0].plot(xvals, yvals, label=label, linestyle='dashed', color='red')
	elif key == 'is':
		label = 'EI'
		axs[0].plot(xvals, yvals, label=label, linestyle='dashdot', color='gold')
	elif key != (2,3):
		continue
	else:
		label = str(key[0] * key[1]) + " particles"
		axs[0].plot(xvals, yvals, label=label)
axs[0].set_title("Action Accuracy by Time")
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("Episode %")
# axs[0].set_ylim(0, 1)
axs[0].tick_params(length=0)
axs[0].tick_params(axis="x")
axs[0].set_xticks(range(10))
axs[0].set_xticklabels(x_ticks)
axs[0].legend()
general.save_fig(fig, f'save/examples/accuracyEpPct.png', tight_layout_kwargs={"rect": [0, 0, 1, 1]})


# parse stored data for plotting
randomParticles = {}
randomKLs = {}
particles = {}
pKLs = {}
for key, accuracies in accuracyDictList.items():
	if key == 'is' or key == 'nn':
		continue
	if len(accuracies) == 0:
		continue
	acc = np.mean(accuracies)
	L2_sample, L1_sample = key
	kl = np.mean(klDict[(L2_sample, L1_sample)])
	if L2_sample == 'random':
		if L1_sample in randomParticles:  # get average of common particles different layers
			randomParticles[L1_sample * 100 / 72] = (randomParticles[L1_sample] + acc) / 2
			randomKLs[L1_sample * 100/ 72] = (randomKLs[L1_sample] + kl) / 2
		else:
			randomParticles[L1_sample* 100 / 72] = acc
			randomKLs[L1_sample* 100 / 72] = kl
	else:
		key = (L2_sample * L1_sample)* 100 / 72  # show percent of full IS
		if key in particles:
			particles[key] = (particles[key] + acc) / 2
			pKLs[key] = (pKLs[key] + kl) / 2
		else:
			particles[key] = acc
			pKLs[key] = kl
keys = []
vals = []
kls = []
rvals = []
rkls = []
for k, v in particles.items():
	keys.append(k)
	vals.append(v)
	kls.append(pKLs[k])
	rvals.append(randomParticles[k])
	rkls.append(randomKLs[k])

nnVals = [np.mean(accuracyDictList["nn"])] * len(keys) 
isVals = [np.mean(accuracyDictList['is'])] * len(keys)
nnKL = [np.mean(nn_kl_div)] * len(keys)

# accuracy and kl divergence plots
num_cols = 2
num_rows = 1
fig, axss = plt.subplots(
	num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), squeeze=False
)
axs = axss.flatten()

ax = axs[0]
ax.plot(keys, isVals, linestyle='dashdot', color='gold', label='EI (72 particles)') 
ax.plot(keys, nnVals, label='NN', color = 'red', linestyle='dashed') 
ax.scatter(keys, vals, color='blue')
ax.plot(keys, vals, label='Our Approach w/ NN', color='blue')
ax.scatter(keys, rvals, color='green')
ax.plot(keys, rvals, label='Our Approach w/o NN', color='green')
ax.set_ylabel("Accuracy")
ax.set_xlabel("% of hypothesis space evaluated")
ax.set_ylim(0, 1)
ax.legend()

ax = axs[1]
ax.plot(keys, nnKL, label='NN', color = 'red', linestyle='dashed') 
ax.scatter(keys, kls, color='blue')
ax.plot(keys, kls, label='Our Approach w/ NN', color='blue')
ax.scatter(keys, rkls, color='green')
ax.plot(keys, rkls, label='Our Approach w/o NN', color='green')
ax.set_ylabel("KL Divergence")
ax.set_xlabel("% of hypothesis space evaluated")
ax.legend()


general.save_fig(fig, f'save/examples/accuracyHypPct.png', tight_layout_kwargs={"rect": [0, 0, 1, 1]})
