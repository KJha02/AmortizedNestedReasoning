import uuid
import numpy as np
import scipy.special
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import imageio
import shutil
import pdb

# Plotting
def save_fig(fig, path, dpi=100, tight_layout_kwargs={}):
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout(**tight_layout_kwargs)
	fig.savefig(path, bbox_inches="tight", dpi=dpi)
	print("Saved to "+ str(path))
	plt.close(fig)


def make_gif(img_paths, gif_path, fps):
	Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
	images = []
	for img_path in tqdm(img_paths):
		images.append(imageio.imread(img_path))
	imageio.mimsave(gif_path, images, duration=1 / fps)
	print("Saved to " + str(gif_path))


def one_hot_to_int(one_hot):
	"""Convert a one-hot vector to an int
	Args
		one_hot [dim]

	Returns (int)
	"""
	assert len(one_hot.shape) == 1
	return (one_hot * torch.arange(len(one_hot), device=one_hot.device)).sum().item()


def get_one_ids(x):
	"""Get a list of ints for which x[i] is one or greater (goal block)
	Args
		x [dim]

	Returns (list of ints)
	"""
	return list(torch.where(x >= 1)[0].cpu().numpy())



def get_tmp_dir():
	temp_dir = "tmp" + str(str(uuid.uuid4())[:8])
	return temp_dir



def plot_scenario1_snapshot(
	img_path, gui_path, agent1_goal_distrib,
	agent1_action_distrib, agent1_goal_int, agent2_goal_distrib, agent2_action_distrib, agent2_goal_int,
	agent3_goal_distrib, agent3_action_distrib, agent3_goal_int, 
):
	num_cols = 4
	num_rows = 2
	fig, axss = plt.subplots(
		num_rows, num_cols, figsize=(num_cols * 5, num_rows * 6), squeeze=False
	)
	axs = axss.flatten()

	# plot the state in first cell
	ax = axs[0]
	im = plt.imread(gui_path)
	ax.imshow(im)


	# plot agent 1 goal inference probability
	ax = axs[1]
	goal_list = ["forward", "left", "right"]
	if agent1_goal_distrib is not None:
		if np.argmax(agent1_goal_distrib) == agent1_goal_int:
			bar_color = "red"
		else:
			bar_color = "C0"
		ax.bar(np.arange(len(goal_list)), agent1_goal_distrib, color=bar_color)
		ax.set_title(f"Belief about Agent 1's Overall Goal")
	else:
		ax.set_title(f"Belief about Agent 1's Overall Goal (N/A)")

	# - Formatting
	ax.set_ylim(0, 1)
	ax.set_yticks([0, 1])
	ax.tick_params(length=0)
	ax.tick_params(axis="x", labelrotation=90)
	ax.set_xticks(range(len(goal_list)))
	ax.set_xticklabels(goal_list)
	ax.tick_params(axis='x', which='both', pad=50)
	


	# plot agent 2 goal inference probability
	ax = axs[2]
	if agent2_goal_distrib is not None:
		if np.argmax(agent2_goal_distrib) == agent2_goal_int:
			bar_color = "red"
		else:
			bar_color = "C0"
		ax.bar(np.arange(len(goal_list)), agent2_goal_distrib, color=bar_color)
		ax.set_title(f"Belief about Agent 2's Overall Goal")
	else:
		ax.set_title(f"Belief about Agent 2's Overall Goal (N/A)")

	# - Formatting
	ax.set_ylim(0, 1)
	ax.set_yticks([0, 1])
	ax.tick_params(length=0)
	ax.tick_params(axis="x", labelrotation=90)
	ax.set_xticks(range(len(goal_list)))
	ax.set_xticklabels(goal_list)
	ax.tick_params(axis='x', which='both', pad=50)

	# plot agent 3 goal inference probability
	ax = axs[3]
	if agent3_goal_distrib is not None:
		if np.argmax(agent3_goal_distrib) == agent3_goal_int:
			bar_color = "red"
		else:
			bar_color = "C0"
		ax.bar(np.arange(len(goal_list)), agent3_goal_distrib, color=bar_color)
		ax.set_title(f"Belief about Agent 3's Overall Goal")
	else:
		ax.set_title(f"Belief about Agent 3's Overall Goal (N/A)")

	# - Formatting
	ax.set_ylim(0, 1)
	ax.set_yticks([0, 1])
	ax.tick_params(length=0)
	ax.tick_params(axis="x", labelrotation=90)
	ax.set_xticks(range(len(goal_list)))
	ax.set_xticklabels(goal_list)
	ax.tick_params(axis='x', which='both', pad=50)


	# plot agent 1 action inference probability
	ax = axs[5]
	action_list = ["forward", "left", "right", "stop", "signal"]
	if agent1_action_distrib is not None:
		bar_color = "C0"
		ax.bar(np.arange(len(action_list)), agent1_action_distrib, color=bar_color)
		ax.set_title(f"Belief about Agent 1's Next Actions")
	else:
		ax.set_title(f"Belief about Agent 1's Next Actions (N/A)")

	# - Formatting
	ax.set_ylim(0, 1)
	ax.set_yticks([0, 1])
	ax.tick_params(length=0)
	ax.tick_params(axis="x", labelrotation=90)
	ax.set_xticks(range(len(action_list)))
	ax.set_xticklabels(action_list)
	ax.tick_params(axis='x', which='both', pad=50)


	# plot agent 2 action inference probability
	ax = axs[6]
	if agent2_action_distrib is not None:
		bar_color = "C0"
		ax.bar(np.arange(len(action_list)), agent2_action_distrib, color=bar_color)
		ax.set_title(f"Belief about Agent 2's Next Actions")
	else:
		ax.set_title(f"Belief about Agent 2's Next Actions (N/A)")

	# - Formatting
	ax.set_ylim(0, 1)
	ax.set_yticks([0, 1])
	ax.tick_params(length=0)
	ax.tick_params(axis="x", labelrotation=90)
	ax.set_xticks(range(len(action_list)))
	ax.set_xticklabels(action_list)
	ax.tick_params(axis='x', which='both', pad=50)

	# plot agent 3 action inference probability
	ax = axs[7]
	if agent3_action_distrib is not None:
		bar_color = "C0"
		ax.bar(np.arange(len(action_list)), agent3_action_distrib, color=bar_color)
		ax.set_title(f"Belief about Agent 3's Next Actions")
	else:
		ax.set_title(f"Belief about Agent 3's Next Actions (N/A)")

	# - Formatting
	ax.set_ylim(0, 1)
	ax.set_yticks([0, 1])
	ax.tick_params(length=0)
	ax.tick_params(axis="x", labelrotation=90)
	ax.set_xticks(range(len(action_list)))
	ax.set_xticklabels(action_list)
	ax.tick_params(axis='x', which='both', pad=50)

	# Format figure
	suptitle_kwargs = {"fontsize": 16}
	fig.suptitle(f"Agent 1 goal: ({goal_list[agent1_goal_int]}) \n Agent 2 goal: ({goal_list[agent2_goal_int]}) \n Agent 3 goal: ({goal_list[agent3_goal_int]})", **suptitle_kwargs)
	save_fig(fig, img_path, tight_layout_kwargs={"rect": [0, 0, 1, 0.95]})





def make_scenario1_gif(
	gif_path, 
	save_dir, 
	states_raw, 
	agent1_goal_distribs,
	agent1_action_distribs,
	agent1_goal_int,
	agent2_goal_distribs,
	agent2_action_distribs,
	agent2_goal_int,
	agent3_goal_distribs,
	agent3_action_distribs,
	agent3_goal_int,
):
	from visualizer import Visualizer  # only make tkinter visualizers locally or this will crash

	tmp_dir = get_tmp_dir()
	img_paths = []
	assert len(states_raw) == len(agent2_action_distribs)
	for timestep, state in enumerate(states_raw):
		state.visualizer = Visualizer(state.width, state.height, ppm=state.ppm)        
		state.render(save_dir + tmp_dir, timestep)
		tmp_gui_path = f"{save_dir}{tmp_dir}/{timestep}_gui.png"
		tmp_img_path = f"{save_dir}{tmp_dir}/{timestep}.png"
		plot_scenario1_snapshot(
			tmp_img_path,
			tmp_gui_path,
			agent1_goal_distribs[timestep],
			agent1_action_distribs[timestep],
			agent1_goal_int,
			agent2_goal_distribs[timestep],
			agent2_action_distribs[timestep],
			agent2_goal_int,
			agent3_goal_distribs[timestep],
			agent3_action_distribs[timestep],
			agent3_goal_int,
		)
		img_paths.append(tmp_img_path)
	make_gif(img_paths, save_dir + gif_path, fps=3)
	shutil.rmtree(save_dir + tmp_dir)

def make_scenario2_gif(
	gif_path, 
	save_dir, 
	states_raw, 
	agent1_goal_distribs,
	agent1_action_distribs,
	agent1_goal_int,
	agent2_goal_distribs,
	agent2_action_distribs,
	agent2_goal_int,
	agent3_goal_distribs,
	agent3_action_distribs,
	agent3_goal_int,
):
	from visualizer import Visualizer  # only make tkinter visualizers locally or this will crash

	tmp_dir = get_tmp_dir()
	img_paths = []
	assert len(states_raw) == len(agent2_action_distribs)
	for timestep, sa in enumerate(states_raw):
		state, action_dict = sa
		state.visualizer = Visualizer(state.width, state.height, ppm=state.ppm)        
		state.render(save_dir + tmp_dir, timestep)
		tmp_gui_path = f"{save_dir}{tmp_dir}/{timestep}_gui.png"
		tmp_img_path = f"{save_dir}{tmp_dir}/{timestep}.png"
		plot_scenario1_snapshot(
			tmp_img_path,
			tmp_gui_path,
			agent1_goal_distribs[timestep],
			agent1_action_distribs[timestep],
			agent1_goal_int,
			agent2_goal_distribs[timestep],
			agent2_action_distribs[timestep],
			agent2_goal_int,
			agent3_goal_distribs[timestep],
			agent3_action_distribs[timestep],
			agent3_goal_int,
		)
		img_paths.append(tmp_img_path)
	make_gif(img_paths, save_dir + gif_path, fps=3)
	shutil.rmtree(save_dir + tmp_dir)




