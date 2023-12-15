import torch
import torch.optim as optim
import time
import sys
# setting path
sys.path.append('../CARLO')
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point, Line
from scenario import Scenario1, Action, get_partial_states
from car_models.ToMnet_car import ToMnet_car_pred, ToMnet_state_pred, ToMnet_exist_pred
# FORWARD = (0, 5000)
# LEFT = (0.42, -50)
# RIGHT = (-0.9, -170)
# STOP = (0, -1e7)
# SIGNAL = (0, 0)
FORWARD = (0, 4000)
LEFT = (0.5, 0)
RIGHT = (-1.15, -1100)
STOP = (0, -1e7)
SIGNAL = (0, 0)

actionStringDict = {FORWARD:"forward", LEFT:"left", RIGHT:"right", STOP:"stop", SIGNAL:"signal"}
actionIntDict = {FORWARD:0, LEFT:1, RIGHT:2, STOP:3, SIGNAL:4}


def update_network(loss, optimizer, scaler=None, model = None):
	"""update network parameters"""
	if model is not None:
		for param in model.parameters():
			param.grad = None
	else:
		optimizer.zero_grad()
	if scaler is not None:
		loss = loss.double()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
	else:
		loss.backward(retain_graph=True)
		optimizer.step()


def save_model(model, path):
	"""save trained model parameters"""
	torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
	"""load trained model parameters"""
	if device is not None:
		model.load_state_dict(dict(torch.load(path, map_location=device)))
	else:
		model.load_state_dict(dict(torch.load(path)))


def expand_vec(vec_tensor, output_dim):
	"""expand vec tensor spatially"""
	return (
		vec_tensor.repeat(1, output_dim[1] * output_dim[2])
		.view(output_dim[1], output_dim[2], output_dim[0])
		.permute(2, 0, 1)
		.unsqueeze(0)
	)


def expand_batch(batch_tensor, output_dim):
	"""expand batch spatially"""
	batch_size = batch_tensor.shape[0]
	return (
		batch_tensor.repeat(1, output_dim[0] * output_dim[1])
		.view(batch_size, output_dim[0], output_dim[1], output_dim[2])
		.permute(0, 3, 1, 2)
		# .unsqueeze(0)
	)

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

def action_to_string(action):
	return actionStringDict[action.value]


def action_to_one_hot(action):
	action_space = ['forward', 'left', 'right', 'stop', 'signal']
	res = [0] * len(action_space)
	res[action_space.index(action)] = 1
	return res

def one_hot_to_action(one_hot_action):
	action_space = ['forward', 'left', 'right', 'stop', 'signal']
	actionString = action_space[np.argmax(one_hot_action)]
	if actionString == "forward":
		return Action.FORWARD
	elif actionString == "left":
		return Action.LEFT
	elif actionString == "right":
		return Action.RIGHT
	elif actionString == "stop":
		return Action.STOP
	else:
		return Action.SIGNAL


# convert world state, actions to tensor
def state_action_to_joint_tensor(state, actions, device='cpu'):
	'''
	state should be a World object
	actions should be a dictionary of agentID --> action taken as an Action object

	tensor is a [a1_exists, a1_x, a1_y, a1_action_one_hot | ... | an_exists, an_x, an_y, an_action_one_hot | time ]
	'''
	res = []
	num_possible_agents = 16
	num_possible_actions = len(actionStringDict)
	for i in range(num_possible_agents):
		agent = state.agent_exists(i)
		if agent is None:
			exists = 0
			x = 0 
			y = 0 
			heading = 0
			action = action_to_one_hot('forward')  # give an arbitrary action if an agent doesn't exist
		else:
			exists = 1
			x = agent.x 
			y = agent.y
			heading = agent.heading
			action = action_to_one_hot(action_to_string(actions[i]))
		res.append(exists)
		res.append(x)
		res.append(y)
		res.append(heading)
		for a in action:  # add the one hot
			res.append(a)
	res.append(state.t)
	return torch.tensor(res, device=device)

# convert tensor to world state
def joint_sa_tensor_to_state_action(joint_tensor, return_scenario=False):
	# convert tensor to numpy array
	representation = joint_tensor.detach().cpu().numpy()

	# Init the scenario
	scenario = Scenario1()
	state = scenario.w
	state.reset()  # remove all dynamic agents but keep buildings

	state.t = representation[-1]  # we stored time
	
	num_possible_agents = 16
	num_possible_actions = len(actionStringDict)

	single_agent_info_size = 4 + num_possible_actions

	actions = {}

	for i in range(0, len(representation)-1, single_agent_info_size):  # jump in chunks, don't include final time value
		curr = i
		agentID = i // single_agent_info_size
		exists = bool(representation[curr])
		if not exists:
			continue
		curr += 1

		[x, y, heading] = representation[curr: curr+3]
		

		curr += 3

		one_hot = representation[curr:curr+num_possible_actions]
		action = one_hot_to_action(one_hot)
		actions[agentID] = action 

		if i in [0, 1, 2, 3, 8, 9, 10, 11]:
			color = "red"
		else:
			color = "blue"

		agent = Car(Point(x, y), heading, ID=agentID, color=color)
		if action != Action.STOP:
			agent.velocity = Point(220, 0)  # initialize speed,
		state.add(agent)

		actions[i] = action
	if return_scenario:
		return state, actions, scenario
	return state, actions

# convert id pair to tensor
def agent_pair_to_tensor(car1_ID, car2_ID, device='cpu'):
	'''
	car1 and car2 are Car objects
	'''
	return torch.tensor([car1_ID, car2_ID],  device=device)

# convert tensor to id pair
def joint_pair_tensor_to_IDs(tensor_pair):
	return tensor_pair.detach().cpu().numpy()


def init_statePred(args, device='cpu'):
	state_model = ToMnet_state_pred(hidden_dim=args.hidden_dim)
	state_model.to(device)
	state_optimizer = optim.Adam(state_model.parameters(), lr=args.lr)
	return state_model, state_optimizer

def init_existPred(args, device='cpu'):
	exist_model = ToMnet_exist_pred(hidden_dim=args.hidden_dim)
	exist_model.to(device)
	exist_optimizer = optim.Adam(exist_model.parameters(), lr=args.lr)
	return exist_model, exist_optimizer

def load_belief_checkpoint(path, device, num_tries=3, exist_model=False):
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
	if exist_model:
		model, optimizer = init_existPred(args, device)
	else:
		model, optimizer = init_statePred(args, device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	stats = checkpoint["stats"]
	return model, optimizer, stats, args