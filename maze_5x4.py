import Q_learning_functions as Q

# ── Maze setup ─────────────────────────────────────────────────────────────
# This file builds the small maze (walls, rewards, starting Q-values) and saves them to JSON files.
rows, cols = 4, 5
start = (1, 4)    # where the agent begins (column, row), counting from 1
goal  = (5, 4)    # where the agent must reach (column, row), counting from 1

# Every cell listed in this set is a wall the agent cannot walk through
walls = {(3, 3), (3, 4)}

# Reward values and learning settings are read from parameter_log.txt (edit them there)
P = Q.PARAMS
R_wall = P["R_wall"]
R_cell = P["R_cell"]
R_goal = P["R_goal"]

# Q-table: one list of 4 Q-values per open cell, in the order [up, down, right, left]
# The key is (column, row). Every action starts at 0.0
maze_Q = {}
for i in range(1, cols+1):
    for j in range(1, rows+1):
        if (i, j) not in walls:
            maze_Q[(i, j)] = [0.0, 0.0, 0.0, 0.0]

# Reward table: every cell gives R_cell, walls give R_wall, and the goal gives R_goal
maze_R = {}
for i in range(1, cols+1):
    for j in range(1, rows+1):
        maze_R[(i, j)] = R_cell
for i in walls:
    maze_R[i] = R_wall
maze_R[goal] = R_goal

# Save the starting rewards and Q-values to JSON files
# The main program reloads them and keeps updating maze_Q.json as training goes on
Q.json_save(maze_Q, "maze_Q.json")
Q.json_save(maze_R, "maze_R.json")

# Learning settings
# gamma = how much future rewards matter, alpha = learning speed
# T = starting temperature (high = very random), it shrinks by T_decay each episode until T_min
gamma, alpha = P["gamma"], P["alpha"]
T, T_decay, T_min = P["T"], P["T_decay"], P["T_min"]
# Size of one maze cell in pixels on screen (bigger than the 50x50 maze, since this maze only has 20 cells)
tile_size = 120