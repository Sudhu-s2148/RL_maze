import pygame
import Q_learning_functions as Q
import time
import math
from GUI_3_0 import (
    lerp_color,
    heat_color,
    draw_rounded_rect,
    compute_q_range,
    draw_arrow,
    build_heatmap_surface,
    draw_frame,
    values,
    TILE_SIZE,
    FPS,
    EPISODE_STEP,
    HUD_H,
    TOTAL_EPISODES,
)

TOTAL_MOVES  = Q.PARAMS["TOTAL_MOVES"]     # max moves allowed per episode before it is cut off (set in parameter_log.txt)
HOLD_SECONDS = Q.PARAMS["HOLD_SECONDS"]    # how long the heatmap stays on screen after each animated episode (set in parameter_log.txt)


# ── Main program ───────────────────────────────────────────────────────────

def main():
    pygame.init()




    # Load the reward of every cell from the JSON file (a dictionary: (column, row) -> reward)
    maze_R = Q.json_load("maze_R.json")
    cols   = max(k[0] for k in maze_R.keys())    # maze width, found from the largest column number
    rows   = max(k[1] for k in maze_R.keys())    # maze height, found from the largest row number
    start, goal = values.start, values.goal
    gamma, alpha = values.gamma, values.alpha

    # Load the starting Q-values: for every cell, one value per action [up, down, right, left]
    # A higher Q-value means that action looks more promising from that cell
    maze_Q = Q.json_load("maze_Q.json")

    # Temperature T controls exploration vs exploitation:
    #   at the start T is high, so the agent picks mostly random moves (exploration)
    #   T shrinks each episode, so the agent increasingly picks the move with the highest
    #   Q-value, i.e. the one most likely to reach the goal (exploitation)
    T, T_decay, T_min = values.T, values.T_decay , values.T_min
    episode    = 0
    tick       = 0        # frame counter, used for the goal's pulsing animation
    running    = True






    # ── (GUI part) ────────────────────────────────────────────────────────────────────────────────────
    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    pygame.display.set_caption("Q-Learning · Gridworld")
    clock  = pygame.time.Clock()

    try:
        font_large = pygame.font.SysFont("Courier New", 13, bold=True)
        font_small = pygame.font.SysFont("Courier New", 10)
    except Exception:
        font_large = pygame.font.SysFont(None, 14)
        font_small = pygame.font.SysFont(None, 11)
    # The heatmap is always shown, and it is rebuilt before each animated episode because the Q-values
    # keep changing as the agent learns. Cells the agent hasn't updated yet keep the regular maze colour
    heatmap_surf = build_heatmap_surface(maze_R, maze_Q, goal, cols, rows)
    q_min, q_max = compute_q_range(maze_Q, maze_R)
    #────────────────────────────────────────────────────────────────────────────────────────────────────






    # ── Training loop: runs for TOTAL_EPISODES episodes ──
    while running and episode < TOTAL_EPISODES:

        # Check if the window was closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Only draw every EPISODE_STEP-th episode, the others train silently (much faster)
        show_this_one = (episode % EPISODE_STEP == 0)
        state         = start
        path_history  = [start]
        moves         = 0




        # One episode: the agent walks through the maze from the start until it
        # reaches the goal or runs out of moves
        while state != goal and moves < TOTAL_MOVES :

            # 1. Softmax turns the 4 Q-values of the current cell into 4 probabilities
            state_prob = Q.softmax(maze_Q[state], T)

            # 2. The agent picks an action using those probabilities
            action     = Q.weighted_random(state_prob)

            # 3. Take the action and update that Q-value with the Bellman equation (Q-learning rule)
            state      = Q.bellmans_update(maze_R, maze_Q, state, action, gamma, alpha, goal)
            path_history.append(state)
            moves += 1





            # Draw the current frame (only for the episodes we chose to show)
            if show_this_one:
                # Check if the window was closed during the animation too
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if not running:
                    break
                tick += 1
                draw_frame(screen, heatmap_surf, maze_R, maze_Q, state,
                           episode, T, path_history, start, goal,
                           cols, rows, font_large, font_small,
                           tick, q_min, q_max)
                clock.tick(FPS)

        # After an animated episode, update the heatmap with what was just learned and keep it on screen
        # for HOLD_SECONDS so you have time to look at it (the window stays responsive meanwhile)
        if show_this_one and running:
            heatmap_surf = build_heatmap_surface(maze_R, maze_Q, goal, cols, rows)
            q_min, q_max = compute_q_range(maze_Q, maze_R)
            hold_until = time.time() + HOLD_SECONDS
            while running and time.time() < hold_until:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                tick += 1
                draw_frame(screen, heatmap_surf, maze_R, maze_Q, state,
                           episode, T, path_history, start, goal,
                           cols, rows, font_large, font_small,
                           tick, q_min, q_max)
                clock.tick(FPS)






        episode += 1
        
        # Lower the temperature a little after every episode:
        # random moves early on, mostly best-known moves later
        T = max(T * T_decay, T_min)





        # Rebuild the heatmap image just before an animated episode, since the Q-values have changed
        if episode % EPISODE_STEP == 0:
            heatmap_surf = build_heatmap_surface(maze_R, maze_Q, goal, cols, rows)
            q_min, q_max = compute_q_range(maze_Q, maze_R)
        if state == goal:
            print("reached goal in ep", episode)

        # Save the Q-values every 50 episodes so progress is not lost
        if episode % 50 == 0:
            Q.json_save(maze_Q, "maze_Q.json")






    # After training: keep the final frame on screen until you close the window.
    # The loop keeps the window responsive and the goal pulsing while you look at the heatmap
    if running:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            tick += 1
            draw_frame(screen, heatmap_surf, maze_R, maze_Q, state,
                       episode, T, path_history, start, goal,
                       cols, rows, font_large, font_small,
                       tick, q_min, q_max)
            clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()