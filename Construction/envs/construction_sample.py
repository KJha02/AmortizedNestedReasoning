import pdb
import random
import numpy as np
import itertools
import envs.construction as construction
import pickle

block_symbols = ['=', 'x', '+', '!', '@', '#', '$', '%', '^', '&']
ALL_BLOCK_PAIRS = construction.ALL_BLOCK_PAIRS
max_possible_block_pairs = len(ALL_BLOCK_PAIRS)
block_pairs = ALL_BLOCK_PAIRS[:max_possible_block_pairs]
ALL_UTILITIES = [0 for i in range(len(ALL_BLOCK_PAIRS))]
highest_util = random.randint(0, len(ALL_BLOCK_PAIRS) - 1)
ALL_UTILITIES[0] = 100
# utilities_permutations = itertools.permutations(ALL_UTILITIES[:max_possible_block_pairs], max_possible_block_pairs)
# util_perm = [util for util in utilities_permutations]
# for util in utilities_permutations:
#     if util not in util_perm:
#         util_perm.append(util)
util_perm = [0] * max_possible_block_pairs

def sample_cell_values(num_rows, num_cols):
    cell_values = []
    for row in range(num_rows):
        value = ""
        for col in range(num_cols):
            if row == 0 or col == 0 or row == num_rows-1 or col == num_cols-1:
                value += '*'
            else:
                value += '.'
        cell_values.append(value)
    return cell_values


def sample_block_pair_utilities(num_possible_block_pairs, prior=None, return_prob=False):
    assert num_possible_block_pairs == max_possible_block_pairs
    final_util_perms = [0] * num_possible_block_pairs
    if prior is not None:
        idx = np.arange(len(final_util_perms))
        utilities_id = np.random.choice(idx, p=prior)
        # utilities = final_util_perms[utilities_id]
        prob = prior[utilities_id]
    else:
        idx = np.arange(len(final_util_perms))
        utilities_id = np.random.choice(idx)
        # utilities = final_util_perms[utilities_id]
        prob = 1.0 / len(final_util_perms)
    final_util_perms[utilities_id] = 100
    utilities = tuple(final_util_perms)
    if return_prob:
        return dict(zip(block_pairs, utilities)), prob
    else:
        return dict(zip(block_pairs, utilities))

def sample_seek_conflict_value(prior=None, return_prob=False):
    seek_conflict_values = [True, False]
    if prior is not None:
        idx = np.arange(len(seek_conflict_values))
        conflict_id = np.random.choice(idx, p=prior)
        seek_conflict = seek_conflict_values[conflict_id]
        prob = prior[conflict_id]
    else:
        idx = np.arange(len(seek_conflict_values))
        conflict_id = np.random.choice(idx)
        seek_conflict = seek_conflict_values[conflict_id]
        prob = 1.0 / len(seek_conflict_values)
    if return_prob:
        return seek_conflict, prob
    else:
        return seek_conflict

def get_free_cells(cell_values):
    free_cells = []
    for row in range(len(cell_values)):
        for col in range(len(cell_values[0])):
            if cell_values[row][col] == '.':
                free_cells.append((row, col))
    return free_cells


def far_apart(colored_block_locations):
    if len(colored_block_locations) < 2:
        return True
    for block1 in colored_block_locations:
        x1, y1 = block1
        for block2 in colored_block_locations:
            if block1 != block2:
                x2, y2 = block2
                if ((y2 - y1)**2 + (x2 - x1)**2)**0.5 <= 1:
                    return False
    return True

def sample_colored_block_locations(cell_values, num_colored_blocks):
    free_cells = get_free_cells(cell_values)
    colored_block_locations = []
    while len(colored_block_locations) < num_colored_blocks:
        block_loc = random.choice(free_cells)
        if block_loc not in colored_block_locations:
            colored_block_locations.append(block_loc)
            if not far_apart(colored_block_locations):
                colored_block_locations.pop()
    return colored_block_locations


def draw_colored_blocks(cell_values, colored_block_locations):
    new_cell_values = cell_values.copy()
    colored_blocks = {}
    for i, loc in enumerate(colored_block_locations):
        row, col = loc
        block = construction.ALL_COLORED_BLOCKS[i]
        construction.strModify(new_cell_values, row, col, block)
        colored_blocks[block] = (row, col)
    return new_cell_values, colored_blocks


def sample_agent_location(cell_values):
    return random.choice(get_free_cells(cell_values))


def colored_block_locations(cell_values):
    colored_blocks = {}
    for row in range(len(cell_values)):
        for col in range(len(cell_values[0])):
            if cell_values[row][col] in block_symbols:
                colored_blocks[cell_values[row][col]] = (row, col)
    return colored_blocks

def sample_construction_env(num_rows=20, num_cols=20,
                            num_possible_block_pairs=max_possible_block_pairs,
                            num_colored_blocks=len(block_symbols)):
    cell_values = sample_cell_values(num_rows, num_cols)
    colored_block_loc = sample_colored_block_locations(cell_values, num_colored_blocks)
    cell_values, colored_blocks = draw_colored_blocks(cell_values, colored_block_loc)
    colored_block_utilities = sample_block_pair_utilities(num_possible_block_pairs)

    # second_agent_location = sample_agent_location(cell_values)
    # construction.strModify(cell_values, second_agent_location[0], second_agent_location[1], '▲')
    agent_location = sample_agent_location(cell_values)

    # Create construction_env
    gridworld = construction.Gridworld(cell_values)
    initial_state = construction.State(gridworld, agent_location, colored_blocks)
    construction_env = construction.ConstructionEnv(initial_state, colored_block_utilities)

    return construction_env

def sample_construction_env_L1(
    num_colored_blocks, num_possible_block_pairs, num_rows=10, num_cols=10, beta_L0=0.01
):
    # Sample L0 env
    construction_env = sample_construction_env(num_rows, num_cols, num_possible_block_pairs, num_colored_blocks)

    # seek_conflict = random.choice([True, False])

    seek_conflict = True
    base_colored_block_utilities_L1 = sample_block_pair_utilities(num_possible_block_pairs, return_prob=False)

    # seek_conflict = False
    # # Sample until there ISN'T a clash
    # done = False
    # while not done:
    #     base_food_truck_utilities_L1 = sample_food_truck_utilities(
    #         num_possible_food_trucks, return_prob=False
    #     )
    #     if not envs.food_trucks.food_truck_utilities_clash(
    #         food_truck_env.food_truck_utilities, base_food_truck_utilities_L1
    #     ):
    #         done = True
    return construction.ConstructionEnvL1(
        seek_conflict,
        base_colored_block_utilities_L1,
        construction_env.initial_state,
        construction_env.colored_block_utilities,
        beta_L0=beta_L0,
    )

def sample_multi_agent_env(
    num_colored_block_locations=10, num_possible_block_pairs=45, seek_conflict=True, num_rows=20, num_cols=20
):
    # Create block pair utilities
    colored_block_utilities_0 = sample_block_pair_utilities(
        num_possible_block_pairs, return_prob=False
    )
    colored_block_utilities_1 = sample_block_pair_utilities(
        num_possible_block_pairs, return_prob=False
    )
    colored_block_utilities = {
        0: colored_block_utilities_0,
        1: colored_block_utilities_1,
    }

    # Create initial state
    # - Create gridworld
    cell_values = sample_cell_values(num_rows, num_cols)
    colored_block_locations = sample_colored_block_locations(cell_values, num_colored_block_locations)
    cell_values, colored_blocks = draw_colored_blocks(cell_values, colored_block_locations)
    gridworld = construction.Gridworld(cell_values)

    # - Create agent locations
    agent_location_0 = sample_agent_location(cell_values)
    agent_location_1 = agent_location_0
    while agent_location_1 == agent_location_0:
        agent_location_1 = sample_agent_location(cell_values)
    agent_locations = {0: agent_location_0, 1: agent_location_1}

    # - Make the initial state
    initial_state_multi_agent = construction.StateMultiAgent(
        gridworld, agent_locations, colored_blocks,
    )

    return construction.ConstructionMultiAgentEnv(initial_state_multi_agent, colored_block_utilities, seek_conflict=seek_conflict)


def default_construction_env():
    chosen = random.choice(g)
    gridworld = construction.Gridworld(
                [
                "********************",
                "*.....^............*",
                "*.................&*",
                "*..................*",
                "*...+..............*",
                "*..................*",
                "*....=.....#.......*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*.............$....*",
                "*..............%...*",
                "*!...........x.....*",
                "*.@................*",
                "*..................*",
                "*..................*",
                "********************"
                ]
            )

    colored_block_utilities_0 = sample_block_pair_utilities(
        45, return_prob=False
    )
    for util in colored_block_utilities_0:
        if util == ("x", "=") or util == ("=", "x"):
            colored_block_utilities_0[util] = 100
        else:
            colored_block_utilities_0[util] = 0

    colored_blocks = {}

    for i in range(len(gridworld.map)):
        for j in range(len(gridworld.map[0])):
            if gridworld.map[i][j] not in ["*", "."]:
                colored_blocks[gridworld.map[i][j]] = (i,j)

    agent_location_0 = (colored_blocks['x'][0], colored_blocks['x'][1] - 1)

    # - Make the initial state
    initial_state = construction.State(
        gridworld, agent_location_0, colored_blocks,
    )
    return construction.ConstructionEnv(initial_state, colored_block_utilities_0)

# def default_multi_agent_env(seek_conflict=True):
#     gridworld = construction.Gridworld(
#                 [
#                     "**********",
#                     "*........*",
#                     "*........*",
#                     "*........*",
#                     "*....x...*",
#                     "*........*",
#                     "*........*",
#                     "*.+....=.*",
#                     "*........*",
#                     "**********"
#                 ]
#             )
#     colored_block_utilities_0 = sample_block_pair_utilities(
#         3, return_prob=False
#     )
#     for util in colored_block_utilities_0:
#         if util == ("x", "+") or util == ("+", "x"):
#             colored_block_utilities_0[util] = 100
#         else:
#             colored_block_utilities_0[util] = 0
#     colored_block_utilities_1 = colored_block_utilities_0
#     colored_block_utilities = {
#         0: colored_block_utilities_0,
#         1: colored_block_utilities_1,
#     }
#     colored_blocks = {
#                 "x": (4, 5),
#                 "+": (7, 2),
#                 "=": (7, 7),
#             }

#     agent_location_0 = (1, 5)
#     agent_location_1 = (8, 5)
#     agent_locations = {0: agent_location_0, 1: agent_location_1}

#     # - Make the initial state
#     initial_state_multi_agent = construction.StateMultiAgent(
#         gridworld, agent_locations, colored_blocks,
#     )

#     return construction.ConstructionMultiAgentEnv(initial_state_multi_agent, colored_block_utilities, seek_conflict=seek_conflict)

def default_multi_agent_env(seek_conflict=True):
    '''
    gridworld = construction.Gridworld(
                [
                    "**********",
                    "*........*",
                    "*........*",
                    "*........*",
                    "*....x...*",
                    "*........*",
                    "*........*",
                    "*.+....=.*",
                    "*........*",
                    "**********"
                ]
            )
    colored_block_utilities_0 = sample_block_pair_utilities(
        3, return_prob=False
    )
    for util in colored_block_utilities_0:
        if util == ("x", "+") or util == ("+", "x"):
            colored_block_utilities_0[util] = 100
        else:
            colored_block_utilities_0[util] = 0
    colored_block_utilities_1 = colored_block_utilities_0
    colored_block_utilities = {
        0: colored_block_utilities_0,
        1: colored_block_utilities_1,
    }
    colored_blocks = {
                "x": (4, 5),
                "+": (7, 2),
                "=": (7, 7),
            }
    agent_location_0 = (1, 5)
    agent_location_1 = (8, 5)
    agent_locations = {0: agent_location_0, 1: agent_location_1}
    '''
    gridworld = construction.Gridworld(
                [
                "********************",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..=..x..+..!..@...*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..................*",
                "*..#..$..%..^..&...*",
                "*..................*",
                "*..................*",
                "*..................*",
                "********************"
                ]
            )
    colored_block_utilities_0 = sample_block_pair_utilities(
        45, return_prob=False
    )
    b1 = random.choice(['=', 'x', '+', '!', '@'])
    b2 = random.choice(['#', '$', '%', '^', '&'])
    for util in colored_block_utilities_0:
        if util == (b1, b2) or util == (b2, b1):
            colored_block_utilities_0[util] = 100
        else:
            colored_block_utilities_0[util] = 0
    colored_blocks = {}
    colored_block_utilities_1 = colored_block_utilities_0
    colored_block_utilities = {
        0: colored_block_utilities_0,
        1: colored_block_utilities_1,
    }
    for i in range(len(gridworld.map)):
        for j in range(len(gridworld.map[0])):
            if gridworld.map[i][j] not in ["*", "."]:
                colored_blocks[gridworld.map[i][j]] = (i,j)
    agent_location_0 = (1, random.randint(1, 18))
    agent_location_1 = (18, random.randint(1, 18))
    agent_locations = {0: agent_location_0, 1: agent_location_1}
    # - Make the initial state
    initial_state_multi_agent = construction.StateMultiAgent(
        gridworld, agent_locations, colored_blocks,
    )
    return construction.ConstructionMultiAgentEnv(initial_state_multi_agent, colored_block_utilities, seek_conflict=seek_conflict)
