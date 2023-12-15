import pdb
import time
import envs.construction_sample
import agents.construction_agent_L0
import copy
import pickle
from tqdm import tqdm


def sample_L0_rollout(env, beta=0.1, useBFS=False):
    """Samples a rollout of an L0 agent

    Args
        env
        beta

    Returns
        rollout: list where each element rollout[i] is a tuple (action, state, obs, reward)
        done (bool)
    """
    rollout = []
    time_limit = 20

    agent = agents.construction_agent_L0.AgentL0(
        env.state.gridworld, env.colored_block_utilities, env.transition, beta
    )

    cumulative_reward, timestep = 0, 0
    state = pickle.loads(pickle.dumps(env.state))
    obs, reward, done = env.reset(), 0, False

    # rollout.append((action, state, obs, reward))
    # while not done:
    for timestep in tqdm(range(time_limit)):
        # start_time = time.time()
        action = agent.get_action(obs, useBFS=useBFS)  # use bfs for actual actions and heuristic for inference
        # if time.time() - start_time > 5:
        #     exit(0)
        next_obs, reward, done, info = env.step(action)
        next_state = env.state
        # try:
        #     assert next_obs == env.transition(obs, action)
        # except:
        #     pdb.set_trace()
        rollout.append((action, state, obs, reward))
        obs = next_obs
        state = next_state
        cumulative_reward += reward
        timestep += 1
        # if timestep == time_limit:
        #     break
        if done:
            break
        # print(f"Rollout is {done}")
        # print(
        #     f"t = {timestep} | action = {action} | reward = {reward} | "
        #     f"cumulative_reward = {cumulative_reward} | done = {done}"
        # )
    # rollout.append((envs.construction.Action.PUT_DOWN, state, obs, reward))
    return rollout, done

if __name__ == "__main__":
    env = envs.construction_sample.sample_construction_env()
    L0_rollout = sample_L0_rollout(env, beta=0.01)
    actions = [roll[0] for roll in L0_rollout[0]]
    states = [roll[1] for roll in L0_rollout[0]]
    for i, state in enumerate(states):
        print(state)
        print(actions[i])
        print("---------")
    print(env.colored_block_utilities)
