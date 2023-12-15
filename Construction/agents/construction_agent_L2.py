import pdb

import numpy as np
import random
import itertools
import copy
import envs
from envs import construction
import agents
from agents import construction_agent_L1
from agents import construction_agent_L0
import pickle
import scipy.special
import test_reasoning_about_construction_L1

NOOP = "NOOP"

def get_colored_block_utilitiess(num_possible_block_pairs):
    # Create an ordered list of colored_block_utilities
    # TODO: merge this with the snippet in test_desire_pred.get_prob_from_inference_output
    final_util_perms = []
    for i in range(num_possible_block_pairs):
        util = [0] * num_possible_block_pairs
        util[i] = 100
        final_util_perms.append(util) 
    return [
        tuple(sorted(zip(construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities,)))
        for utilities in final_util_perms
    ]

class AgentL2:
    """Level-2 agent

    Args
        colored_block_utilities (dictionary)
            key (tuple (str, str)) is the block pair
            value (int) is the utility of that block pair

            Contains utilities for all possible food trucks in the map
            len(block_pair_utilities) can be > number of colored blocks in the gridworld
        my_initial_state (State)
        other_agent_initial_location (tuple of ints): location of the other agent
        my_transition
        other_agent_transition
        other_agent_inference_algorithm
        other_agent_beta_L0
        other_agent_beta_L1
        other_agent_num_samples
        other_agent_model
    """

    def __init__(
        self,
        colored_block_utilities,
        my_initial_state,
        other_agent_initial_location,
        my_transition,
        other_agent_transition,
        other_agent_inference_algorithm,
        other_agent_beta_L0,
        other_agent_beta_L1,
        num_samples=2,
        other_agent_num_samples=5,
        other_agent_model=None,
        inference_algorithm="IS",
        beta_L2=10.0,
        comb_weight=25,
        prior=0.5,
        entropy_weight=50,
        useBFS=False
    ):
        self.my_current_states = []
        # This will always have one fewer elements than my_previous_states
        self.my_previous_actions = []
        self.colored_block_utilities = colored_block_utilities
        self.my_initial_state = my_initial_state

        self.num_samples = num_samples

        self.other_agent_initial_location = other_agent_initial_location
        self.other_agent_initial_inv = self.my_initial_state.block_picked
        self.entropy_weight= entropy_weight

        self.states_L1 = []
        self.actions_L1 = []

        self.my_transition = my_transition
        self.my_inference_algorithm = inference_algorithm
        self.other_agent_transition = other_agent_transition
        self.other_agent_inference_algorithm = other_agent_inference_algorithm
        self.beta_L2 = beta_L2
        self.other_agent_beta_L0 = other_agent_beta_L0
        self.other_agent_beta_L1 = other_agent_beta_L1
        self.other_agent_num_samples = other_agent_num_samples
        self.other_agent_model = other_agent_model
        self.comb_weight = comb_weight
        self.prior = prior

        self.useBFS = useBFS

        self.num_possible_block_pairs = len(self.colored_block_utilities)

        # Initialize the other agent
        seek_conflict = prior >= 0.5  # Only do this with a conflict-seeking other agent

        self.base_colored_block_utilities_L1 = self.colored_block_utilities

        self.other_agent = construction_agent_L1.AgentL1(
            seek_conflict,
            self.base_colored_block_utilities_L1,
            self.num_possible_block_pairs,
            self.my_initial_state,
            self.other_agent_initial_location,
            self.my_transition,
            self.other_agent_transition,
            self.other_agent_inference_algorithm,
            beta_L0=self.other_agent_beta_L0,
            beta_L1=self.other_agent_beta_L1,
            num_samples=self.other_agent_num_samples,
            model=self.other_agent_model,
            ground_truth_colored_block_utilities_L0=self.colored_block_utilities
        )

        # Initialize a Level-0 version of this agent with different block_pair utilities
        self.agents_L0 = {
            colored_block_utilities: construction_agent_L0.AgentL0(
                self.my_initial_state.gridworld,
                dict(colored_block_utilities),
                self.my_transition,
                self.other_agent_beta_L0,
            )
            for colored_block_utilities in get_colored_block_utilitiess(self.num_possible_block_pairs)
        }

        self.particles = None
        self.log_weight = None
        self.prev_beliefs = []

    def num_timesteps(self):
        return len(self.my_current_states) - 1

    def get_belief(self, seek_conflict_proposal_probs=None):
        if self.num_timesteps() == 1:
            assert self.particles is None
            assert seek_conflict_proposal_probs is not None
            # if self.other_agent_inference_algorithm == "SMC" or self.other_agent_inference_algorithm == "IS":
            #     seek_conflict_proposal_probs = None
            env = construction.ConstructionEnvL1(True, self.base_colored_block_utilities_L1,
                                               self.my_initial_state, self.colored_block_utilities,
                                                 self.other_agent_initial_location, self.other_agent_initial_inv, beta_L0=self.other_agent_beta_L0,
                                                 beta_L1=self.other_agent_beta_L1)
            env.timestep = self.num_timesteps()
            self.particles, self.log_weight, L1_action_probss = test_reasoning_about_construction_L1.init_particles(
                env=env,
                data=(self.states_L1[-2], self.actions_L1[-1]),
                num_samples=self.num_samples,
                other_agent_num_samples=self.other_agent_num_samples,
                seek_conflict_proposal_probs=seek_conflict_proposal_probs,
                beta=self.other_agent_beta_L1,
                inference_algorithm=self.other_agent_inference_algorithm,
                model=self.other_agent_model
            )
        else:
            try:
                assert len(self.particles[1][0]) == self.num_timesteps() - 1
            except:
                pdb.set_trace()
            if self.my_inference_algorithm == "SMC":
                resample = True
                rejuvenate = False
                seek_conflict_proposal_probs = None
            elif self.my_inference_algorithm == "IS":
                resample = False
                rejuvenate = False
                seek_conflict_proposal_probs = None
            elif self.my_inference_algorithm == "Online_IS+NN":
                resample = False
                rejuvenate = False
                seek_conflict_proposal_probs = None  # TODO: Once L0 inference amortization works, make this L1 inference
            else:
                raise NotImplementedError(f"{self.other_agent_inference_algorithm} not implemented yet")
            self.particles, self.log_weight, L1_action_probss = test_reasoning_about_construction_L1.update_particles(
                particles=self.particles,
                log_weight=self.log_weight,
                data=(
                    self.states_L1[-2],
                    self.actions_L1[-2],  # consider deleting if we don't use it
                    self.states_L1[-1],
                    self.actions_L1[-1],
                ),
                resample=resample,
                rejuvenate=rejuvenate,
                seek_conflict_proposal_probs=seek_conflict_proposal_probs
            )
        seek_conflictss, beliefss, env_clones, agent_clones = self.particles
        return test_reasoning_about_construction_L1.get_posterior(
            list(zip(seek_conflictss, beliefss)), self.log_weight
        ), L1_action_probss


    def get_action(self, my_current_state=None, my_previous_action=None, return_info=False,
                   current_state_L1=None, prev_action_L1=None):
        """
        Args
            my_current_state (State)
            my_previous_action (Action)
        """
        # Update the other agent's belief
        if my_previous_action == NOOP:
            other_agent_current_belief = self.other_agent.get_belief()
        else:
            if my_current_state is not None:
                self.my_current_states.append(my_current_state)

            if current_state_L1 is not None:
                self.states_L1.append(current_state_L1)
            if prev_action_L1 is not None:
                self.actions_L1.append(prev_action_L1)

            if my_previous_action is not None:
                self.my_previous_actions.append(my_previous_action)

            if my_previous_action is not None:
                self.other_agent.observations.append(
                    construction.ObservationL1(
                        self.my_current_states[-1], my_previous_action
                    )
                )
                other_agent_current_belief = self.other_agent.get_belief()
            else:
                other_agent_current_belief = None

        # Try all actions and see which one confuses the other agent the most
        colored_block_utilitiess = get_colored_block_utilitiess(self.num_possible_block_pairs)
        action_space = list(construction.Action)
        expected_distance = {}
        action_entropy = {}
        next_beliefs = {}
        # pdb.set_trace()
        for action in action_space:
            other_agent_copy = pickle.loads(pickle.dumps(self.other_agent))

            # Compute other agent's next belief

            other_agent_copy.observations.append(
                construction.ObservationL1(self.my_current_states[-1], action)
            )
            # if action == construction.Action.RIGHT:
            #     pdb.set_trace()

            # print(action)
            # print(belief_dict)


            # Compute the expected difference
            # expected_distance[action] = -belief_dict[self.colored_block_utilities]
            expected_distance[action] = 0
            # if self.num_timesteps() > 0:
            # try:
            #     belief = other_agent_copy.get_belief()
            # except:
            #     pdb.set_trace()
            belief = other_agent_copy.get_belief()
            # P(g | a)
            # pdb.set_trace()
            belief_dict = dict(belief)
            next_beliefs[action] = belief
            entropy = 0
            for colored_block_utilities in colored_block_utilitiess:
                if colored_block_utilities in belief_dict:
                    # P(g | a) * dist(g', g)  -- weighing distance between ground truth and this utility by probability of utility
                    block_pair_dict = dict(colored_block_utilities)
                    expected_distance[action] += belief_dict[colored_block_utilities] * distance_3(
                        block_pair_dict, self.colored_block_utilities
                    )
                    prob_g_a = max(belief_dict[colored_block_utilities], 1e-6)
                    entropy += prob_g_a * np.log(prob_g_a)
            action_entropy[action] = -entropy
            # print(f"expected distance for {action}: {expected_distance[action]}")
            # print("------")
        ## TODO: Check that planner and L2 agent are working properly
        # Only consider actions that a Level-0 version of this agent would plausibly take
        # - Create L0 observations from my_current_states
        observations_L0 = [
            construction.ConstructionEnv(state, {}).get_observation()
            for state in self.my_current_states
        ]

        # Ground truth util used for actually completing task
        ground_truth_util = None
        # - Compute L0 action probs for each action and block pair utilities
        action_probs_L0 = np.zeros((len(colored_block_utilitiess), len(action_space)))
        for i, colored_block_utilities in enumerate(colored_block_utilitiess):
            if dict(colored_block_utilities) == self.colored_block_utilities:
                ground_truth_util = colored_block_utilities
            self.agents_L0[colored_block_utilities].observations = observations_L0
            action_probs_L0[i] = self.agents_L0[colored_block_utilities].get_action_probs()
            # print(f"Utilities: {colored_block_utilities} --> action_probs_L0: {action_probs_L0[i]}")
        # - Check which action are plausible
        plausible_actions = np.max(action_probs_L0, axis=0) > 0.05





        # seek_conflict = False
        probIndividual = 0.0  # what's probability L1 has it's own agenda
        probSocial = 1 - probIndividual
        belief_L2 = [(False, (1 - self.prior) * probSocial), (True, self.prior * probSocial)]  # based on prior
        # for b_pair in construction.ALL_BLOCK_PAIRS[:self.num_possible_block_pairs]:
        #     belief_L2.append((b_pair, probIndividual / self.num_possible_block_pairs))

        L1_action_probss = [[0]*len(construction.Action)]*len(belief_L2)
        if self.num_timesteps() > 0:  # initially assume L1 is hurting
        # probabilistically infer if L1 is helping or hurting based on particle inference
            # if self.num_timesteps() == 8 or self.num_timesteps() == 9:
            #     pdb.set_trace()
            prior_prob = [b[1] for b in belief_L2]
            belief_L2, L1_action_probss = self.get_belief(seek_conflict_proposal_probs=prior_prob)
            
        # TODO: comment line 300 for L2 doing inference
        # belief_L2 = [(True, self.prior), (False, 1 - self.prior)]  # based on prior
            # seek_conflict = np.random.choice(conflict, p=conflict_prob)
        conflicts = [False, True]  # + construction.ALL_BLOCK_PAIRS[:self.num_possible_block_pairs]
        conflict_prob = [0.0] * len(conflicts)
        for (c, p) in belief_L2:
            conflict_prob[conflicts.index(c)] = p
        
        self.other_agent.seek_conflict = bool(np.argmax(conflict_prob[:2]))
        # print(f"L2's Belief about L1's intentions at timestep {self.num_timesteps()}: {belief_L2}")




        agent_L0 = self.agents_L0[ground_truth_util]
        (agent_L0_state, prob) = agent_L0.get_belief()[0]
        v_L2 = {a: 0 for a in action_space}
        actions, values = construction_agent_L0.determine_subgoals(state_0=agent_L0_state,
                                                                   transition=agent_L0.transition,
                                                                   colored_block_utilities=agent_L0.colored_block_utilities,
                                                                   useBFS=self.useBFS)
        # actions, values = construction_agent_L0.plan_shortest_path(agent_L0_state,
        #                                                            agent_L0.transition,
        #                                                            agent_L0.colored_block_utilities)

        for i, a in enumerate(actions):
            v_L2[construction.Action(a.value)] += prob * values[i]
        # pdb.set_trace()
        # print(f"L2 value before combination: {v_L2}")
        # print(f"L2 expected distance: {expected_distance}")



        for a, dist in expected_distance.items():
            for i, conflict in enumerate(conflicts):
                c_prob = conflict_prob[i]
                if conflict:
                    v_L2[a] += self.comb_weight * dist * c_prob + self.entropy_weight * c_prob * action_entropy[a]
                else:
                    v_L2[a] -= self.comb_weight * dist * c_prob - self.entropy_weight * c_prob * action_entropy[a]

        # TODO: Add third term to increase entropy of ground truth distributions for helping and decrease it for hurting

        # - Zero out expected distance for actions that are not plausible
        for i, action in enumerate(action_space):
            if not plausible_actions[i]:
                v_L2[action] = -1e6
            # if seek_conflict or dist == -1e6:
            #     v_L2[a] += self.comb_weight * dist  # maximize expected distance if hurting
            # else:
            #     v_L2[a] -= self.comb_weight * dist  # minimize expected distance if helping
        # print(f"Value for each action at L2 level after combination: {v_L2}")

        # pdb.set_trace()

        action_log_probs = np.full((len(action_space),), -1e6)
        for action_id, action in enumerate(action_space):
            if v_L2[action] is not None:
                action_log_probs[action_id] = self.beta_L2 * v_L2[action]
        if scipy.special.logsumexp(action_log_probs) < np.log(1e-6):
            action_probs_normalized = [1/(len(action_space))] * len(action_space)
        else:
            action_log_probs_normalized = action_log_probs - scipy.special.logsumexp(
                action_log_probs
            )
            action_probs_normalized = np.exp(action_log_probs_normalized)
            action_probs_normalized = action_probs_normalized / np.sum(action_probs_normalized)
            if np.isnan(action_probs_normalized).any():
                raise RuntimeError("nan action probs")

        # charlie = 0.3  # probability of choosing a confusing action
        #print(f"L2 action probabilities: {action_probs_normalized}")
        # action = np.random.choice(action_space, p=action_probs_normalized) uncomment for stochastic
        max_actions = []
        max_action_prob = np.max(action_probs_normalized)
        for i, a in enumerate(action_space):
            if action_probs_normalized[i] == max_action_prob:
                max_actions.append(a)
        action = max_actions[np.random.randint(0, len(max_actions))]  # FOR DETERMINISTIC L2

        # Exception:
        # Don't STOP at the goal if I'm on a colored block goal location
        # Do NOOP instead


        map = my_current_state.gridworld.map
        if (
            action == construction.Action.STOP
            and my_current_state.agent_location in my_current_state.colored_blocks.values()
        ):
            curr_block = map[my_current_state.agent_location[0]][my_current_state.agent_location[1]]
            max_block_1, max_block_2 = max(
                self.colored_block_utilities, key=self.colored_block_utilities.get
            )
            x1, y1 = my_current_state.colored_blocks[max_block_1]
            x2, y2 = my_current_state.colored_blocks[max_block_2]
            distanceCheck = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            if distanceCheck and (curr_block == max_block_1 or curr_block == max_block_2):
                print("Did a NOOP")
                action = random.choice(
                    list(set(construction.Action) - set([construction.Action.STOP]))
                )
        temp = pickle.loads(pickle.dumps(self.prev_beliefs))
        self.prev_beliefs.append(dict(belief_L2))
        if return_info:
            return (
                action,
                {
                    "expected_distance": expected_distance,
                    "next_beliefs": next_beliefs,
                    "other_agent_current_belief": other_agent_current_belief,
                    "other_agent_seek_conflict": conflict_prob[:2],
                    "L2_inference": conflict_prob,
                    "prev_other_agent_seek_conflict": temp,
                    "L2_action_probs": dict(zip(list(construction.Action), action_probs_normalized)),
                    "L2_guess_conflict": self.other_agent.seek_conflict,
                    "L1_imagined_probs": L1_action_probss
                },
            )
        else:
            return action

def hamming_distance(string1, string2):
    """From https://en.wikipedia.org/wiki/Hamming_distance#Algorithm_example"""
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter

def colored_block_utilities_to_str(colored_block_utilities):
    """Return block pair names from least to most preferred"""
    sorted_pairs = [kv_[0] for kv_ in sorted(list(colored_block_utilities.items()), key=lambda kv: kv[1])]
    res = ""
    for pair in sorted_pairs:
        res += pair[0] + pair[1]
    return res


def distance(colored_block_utilities_1, colored_block_utilities_2):
    """Distance between two colored block utilities (dicts)"""
    return hamming_distance(
        colored_block_utilities_to_str(colored_block_utilities_1),
        colored_block_utilities_to_str(colored_block_utilities_2),
    )


def get_ordered_pairs(x):
    result = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            temp = "".join([x[i], x[j]])
            reverse = "".join([x[j], x[i]])
            if reverse not in result and x[i] != x[j]:
                result.append(temp)
    return result


def get_num_different_ordered_pairs(x, y):
    pair1 = set(get_ordered_pairs(x))
    pair2 = set(get_ordered_pairs(y))
    inter = pair1.intersection(pair2)
    pdb.set_trace()
    return len(x) - len(set(get_ordered_pairs(x)).intersection(set(get_ordered_pairs(y))))


def distance_2(colored_block_utilities_1, colored_block_utilities_2):
    """Distance between two colored block utilities (dicts)"""
    return get_num_different_ordered_pairs(
        colored_block_utilities_to_str(colored_block_utilities_1),
        colored_block_utilities_to_str(colored_block_utilities_2),
    )

def distance_3(colored_block_utilities_1, colored_block_utilities_2):
    if colored_block_utilities_1 == colored_block_utilities_2:
        return -1
    return 0
