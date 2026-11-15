import math
import random
import json
import ast

import os

# ── Parameters ─────────────────────────────────────────────────────────────
# All settings (gamma, alpha, T, rewards, ...) live in parameter_log.txt, in the same folder as this file.
# This function reads that file so you only ever have to edit the numbers there.
PARAMETER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameter_log.txt")

# Settings the program cannot run without
REQUIRED_PARAMETERS = ["gamma", "alpha", "T", "T_decay", "T_min",
                       "R_wall", "R_cell", "R_goal",
                       "TOTAL_EPISODES", "TOTAL_MOVES", "EPISODE_STEP", "HOLD_SECONDS", "FPS"]

def load_parameters(filename=PARAMETER_FILE):
    params = {}
    with open(filename, "r") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.split("#")[0].strip()    # remove comments
            if "=" not in line:                  # skip blank lines and plain text
                continue
            left, right = line.split("=", 1)
            names = [n.strip() for n in left.split(",")]
            try:
                value = ast.literal_eval(right.strip())
            except (ValueError, SyntaxError):
                raise ValueError(f"{filename}, line {line_number}: could not read the value in '{line}'")
            # one name -> one value, several names -> several values in the same order
            values = value if (len(names) > 1 and isinstance(value, tuple)) else (value,)
            if len(names) != len(values):
                raise ValueError(f"{filename}, line {line_number}: {len(names)} names but {len(values)} values in '{line}'")
            for name, val in zip(names, values):
                params[name] = val
    missing = [name for name in REQUIRED_PARAMETERS if name not in params]
    if missing:
        raise ValueError(f"{filename} is missing: {', '.join(missing)}")
    return params

PARAMS = load_parameters()

# The wall reward is used below to recognise walls
R_wall = PARAMS["R_wall"]






 
# ── Softmax ────────────────────────────────────────────────────────────────
# Turns a list of Q-values into probabilities that add up to 1.
# The temperature T controls how spread out the probabilities are:
#   high T -> almost equal probabilities (agent explores randomly)
#   low T  -> the highest Q-value gets most of the probability (agent exploits what it learned)
def softmax(list_values, T):
    T = max(T, 0.05)    # never let T get too close to 0, otherwise we divide by ~0

    scaled_values = [i / T for i in list_values]

    # subtracting the biggest value keeps math.exp from overflowing, the result stays the same
    max_val = max(scaled_values)
    exp_values = [math.exp(v - max_val) for v in scaled_values]

    sum_of_exp_values = sum(exp_values)
    probabilities = [round(ev / sum_of_exp_values, 3) for ev in exp_values]

    return probabilities





 
# ── Q-learning update (Bellman equation) ───────────────────────────────────
# Moves the agent one step and updates the Q-value of the (state, action) it just used.
#
#   Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a'(Q(s', a')) - Q(s, a) ]
#
#   r + gamma * max_a'(Q(s', a'))  = TD target (what we now think the value should be)
#   Q(s, a)                        = our current prediction
#   target - prediction            = TD error (how wrong we were)
def bellmans_update(rewards, Q_values, state, action, gamma, alpha, goal):

    # s' (next state): where the action takes us. We need it for max_a'(Q(s', a')) in the equation
    new_state = state_updater(state, action, rewards)

    # In this maze the reward depends only on the cell we land in, so we read it from s'
    reward = rewards[new_state]

    # max_a'(Q(s', a')) = the best Q-value available from the next state
    # If s' is the goal the episode ends, so there is no future value and we use 0
    max_Q = 0.0 if new_state == goal else max(Q_values[new_state])

    # TD error = (r + gamma * max_a'(Q(s', a'))) - Q(s, a)
    error = (reward + gamma * max_Q) - Q_values[state][action]

    # Nudge the old Q-value a small step (alpha) towards the target
    Q_values[state][action] += alpha * error

    return new_state





 
# ── Weighted random choice ─────────────────────────────────────────────────
# Picks an action index using the probabilities from softmax.
# We draw a random number between 0 and 1 and see which probability "slice" it lands in.
# Example: probabilities [0.1, 0.6, 0.2, 0.1] -> index 1 is picked about 60% of the time.
def weighted_random(probabilities):
    random_num = random.random()
    cumulative = 0.0
    for i, prob in enumerate(probabilities):
        cumulative += prob
        if random_num < cumulative:
            return i
    return len(probabilities) - 1    # safety net in case rounding leaves a tiny gap







# ── Movement ───────────────────────────────────────────────────────────────
# Works out the new cell after an action.
# Actions: 0 = up, 1 = down, 2 = right, 3 = left
def state_updater(state, action, rewards):
    i, j = state
    if action == 0:
        new_state = (i, j-1)
    elif action == 1:
        new_state = (i, j+1)
    elif action == 2:
        new_state = (i+1, j)
    else:
        new_state = (i-1, j)
    # If the move hits a wall (or leaves the maze) the agent stays in its old cell
    if new_state not in rewards or rewards[new_state] == R_wall:
        return state
    return new_state

 
# ── Saving and loading ─────────────────────────────────────────────────────
# JSON cannot use tuples like (3, 4) as dictionary keys, so we turn keys into text when
# saving and convert them back into tuples when loading.
def json_save(data, filename):
    data_serializable = {str(k): v for k, v in data.items()}
    with open(filename, "w") as fh:
        json.dump(data_serializable, fh)

def json_load(filename):
    with open(filename, "r") as fh:
        loaded = json.load(fh)
        new_data = {ast.literal_eval(k): v for k, v in loaded.items()}
    return new_data