from stable_baselines3.common.env_checker import check_env
from envs.constructionGym import ConstructionGymEnv
import envs.construction_sample
from utils.construction_data import multi_agent_state_to_state_tensor, multi_agent_state_tensor_to_state

# non_gym_env = sample_construction_env()

# gym_env = ConstructionGymEnv(non_gym_env.initial_state, non_gym_env.colored_block_utilities)

# check_env(gym_env, warn=True)

sample = envs.construction_sample.sample_multi_agent_env()
print(sample.state)
s_tensor = multi_agent_state_to_state_tensor(sample.state, 45)
converted_s = multi_agent_state_tensor_to_state(s_tensor, 45)
print(converted_s)

