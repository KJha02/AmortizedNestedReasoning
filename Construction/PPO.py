from envs.constructionGym import ConstructionGymEnv
from envs.construction_sample import sample_construction_env, default_construction_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from utils.construction_data import state_tensor_to_state, block_pair_utilities_to_desire_int
from test_construction_agent_L0 import plot_L0_snapshot
import utils.general
import shutil
import torch
import random
import numpy as np
import pickle
import pdb

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
cuda = torch.cuda.is_available()
if cuda:
	torch.cuda.manual_seed(123)
	device = "cuda"
else:
	device = "cpu"

save_dir = "save/construction/ppo45L0"

gif_save = "save/construction/ppoRes/"


non_gym_env = sample_construction_env()

gym_env = ConstructionGymEnv(non_gym_env.initial_state, non_gym_env.colored_block_utilities)

model = DQN("MlpPolicy", gym_env, verbose=2, gamma=0.95, device=device, learning_rate=0.001)
model.learn(total_timesteps=10000)
# model.save(save_dir)

# del model

# model = PPO.load(save_dir, env=gym_env)


# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(mean_reward)


num_actions = 6
env_copy = pickle.loads(pickle.dumps(gym_env))

for i in range(1):
	gif_path = gif_save + f"{i}.gif"
	img_paths  = []
	img_gui_paths = []
	tmp_dir = utils.general.get_tmp_dir()

	obs = gym_env.reset()

	assert gym_env.colored_block_utilities == env_copy.colored_block_utilities, "Utilities not the same"
	assert gym_env.observation.all() == env_copy.observation.all(), "Observations not the same"

	done = False

	num_possible_block_pairs = len(gym_env.colored_block_utilities)
	rollout_desire_int = block_pair_utilities_to_desire_int(gym_env.colored_block_utilities, num_possible_block_pairs)
	timestep = 0

	total_reward = 0

	while timestep < 40:
		converted_state = state_tensor_to_state(torch.tensor(obs))

		img_path = f"{gif_save}{tmp_dir}/{timestep}.png"
		img_path_gui = f"{gif_save}{tmp_dir}/{timestep}_gui.png"

		empty_inference = [0.0] * len(gym_env.colored_block_utilities)

		plot_L0_snapshot(img_path, 
				img_path_gui,
				converted_state, 
				gym_env.colored_block_utilities, 
				empty_inference, 
				empty_inference,
				empty_inference,
				rollout_desire_int)
		img_paths.append(img_path)
		img_gui_paths.append(img_path_gui)


		if done:
			break

		action, _states = model.predict(obs)
		print(action)
		obs, reward, done, info = gym_env.step(action)
		total_reward += reward
		timestep += 1

	utils.general.make_gif(img_paths, gif_path, 3)
	shutil.rmtree(gif_save + tmp_dir)

	print(f"Total Reward for specific episode {i} is {total_reward}")

