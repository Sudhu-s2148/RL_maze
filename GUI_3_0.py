import pygame
import math
import argparse
import importlib
import Q_learning_functions as Q

MAZES = {
    "5x4": "maze_5x4",
    "20x20-backtracker": "maze_20x20_bactracker",
    "20x20-rooms": "maze_20x20_rooms",
    "50x50-rooms": "maze_50x50_rooms",
}

parser = argparse.ArgumentParser(description="Run Q-learning on a chosen maze")
parser.add_argument("--maze", choices=MAZES.keys(), default="5x4",
                    help="which maze to load (default: %(default)s)")
args = parser.parse_args()

values = importlib.import_module(MAZES[args.maze])

# ── Settings ───────────────────────────────────────────────────────────────
TILE_SIZE      = values.tile_size
FPS            = Q.PARAMS["FPS"]              # all of these are set in parameter_log.txt
EPISODE_STEP   = Q.PARAMS["EPISODE_STEP"]     # only every EPISODE_STEP-th episode is animated
TOTAL_EPISODES = Q.PARAMS["TOTAL_EPISODES"]   # shown in the HUD (the main program imports this too)
PATH_TAIL_LEN  = 100    # how many recent steps to draw in the path trail
R_WALL         = values.R_wall    # reward that marks a wall cell (set in parameter_log.txt)
 
# ── Colour palette ─────────────────────────────────────────────────────────
BG          = (13,  13,  18)
WALL        = (15,  15,  23)
WALL_EDGE   = (40,  40,  55)
CELL        = (30,  30,  42)
CELL_EDGE   = (45,  45,  62)
 
GOAL_FILL   = (20,  180, 100)
GOAL_GLOW   = (20,  220, 110)
GOAL_EDGE   = (10,  100,  55)
 
AGENT_FILL  = (60,  140, 255)
AGENT_GLOW  = (80,  170, 255)
 
PATH_HEAD   = (60,  140, 255)
PATH_TAIL   = (30,   60, 120)
 
HUD_BG      = (18,  18,  26)
HUD_ACCENT  = (60,  140, 255)
HUD_TEXT    = (200, 210, 230)
HUD_DIM     = (90,  100, 120)
HUD_H       = 44        # height of the top info bar in pixels
 
START_FILL  = (255, 160,  40)
 
# ── Heatmap colours ────────────────────────────────────────────────────────
# negative Q-value -> red, around zero -> dark, positive Q-value -> green
HEAT_RED    = (235,  60,  60)
HEAT_ZERO   = (28,   28,  40)
HEAT_GREEN  = (60,  230, 100)
 
 
# Every Q-value starts at this number (set in the values file). Used to spot cells the agent hasn't updated yet
INITIAL_Q = values.maze_Q[values.start][0]
 
 
# ── Helper functions ───────────────────────────────────────────────────────
 
# Returns the best Q-value among the actions the agent has actually tried (Q-value changed from its start value).
# Actions never tried are ignored, so they can't fake a high value.
# Returns None if the agent has never learned anything in this cell
def best_learned_q(q_list):
    learned = [q for q in q_list if q != INITIAL_Q]
    return max(learned) if learned else None
 
 
# Blends colour c1 into colour c2. t = 0 gives c1, t = 1 gives c2, 0.5 gives the middle
def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
 
 
# Converts a Q-value into a number t between -1 and +1 for colouring:
#   t < 0 -> red   (-1 = the most negative Q-value in the maze)
#   t = 0 -> dark  (Q-value of about 0)
#   t > 0 -> green (+1 = the most positive Q-value in the maze)
# Negatives are scaled against q_min and positives against q_max, so zero is always the neutral colour.
# The square root makes small positive values easier to see, so you can watch
# the green slowly spread from the goal towards the start.
def q_to_t(q, q_min, q_max):
    if q >= 0:
        if q_max <= 0:
            return 0.0
        return min(q / q_max, 1.0) ** 0.5
    if q_min >= 0:
        return 0.0
    return -(min(q / q_min, 1.0) ** 0.5)
 
 
# Turns t (from q_to_t) into an actual colour
def heat_color(t):
    t = max(-1.0, min(1.0, t))
    if t >= 0:
        return lerp_color(HEAT_ZERO, HEAT_GREEN, t)
    return lerp_color(HEAT_ZERO, HEAT_RED, -t)
 
 
# Draws a rectangle with rounded corners and an optional border
def draw_rounded_rect(surface, color, rect, radius, border_color=None, border_width=1):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=radius)
 
 
# Finds the lowest and highest "best Q-value" over all open cells that have been updated.
# Untouched cells are skipped. The heatmap uses these two numbers to decide how strong each colour should be
def compute_q_range(maze_Q, maze_R):
    vals = [best_learned_q(q) for s, q in maze_Q.items()
            if maze_R.get(s, -20) != -20 and best_learned_q(q) is not None]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    return (lo, lo + 1.0) if hi == lo else (lo, hi)
 
 
# Draws a small arrow in a cell showing which direction the agent currently prefers.
# Adjust dirs to match your state_updater action mapping:
#   0 = up, 1 = down, 2 = right, 3 = left
def draw_arrow(surface, rect, action_idx, cell_color):
    dirs = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
    if action_idx not in dirs:
        return
    dx, dy  = dirs[action_idx]
    cx, cy  = rect.centerx, rect.centery
    tip     = (cx + dx * TILE_SIZE // 5, cy + dy * TILE_SIZE // 5)
    base    = (cx - dx * TILE_SIZE // 7, cy - dy * TILE_SIZE // 7)
    color   = lerp_color(cell_color, (255, 255, 255), 0.55)
    pygame.draw.line(surface, color, base, tip, 2)
    px, py  = -dy, dx
    h1      = (tip[0] - dx * 5 + px * 4, tip[1] - dy * 5 + py * 4)
    h2      = (tip[0] - dx * 5 - px * 4, tip[1] - dy * 5 - py * 4)
    pygame.draw.polygon(surface, color, [tip, h1, h2])
 
 
# ── Static surface builders ────────────────────────────────────────────────
 
# Draws the plain maze (walls, open cells, goal) onto an image once.
# It never changes during training, so we build it only one time and reuse it
def build_normal_surface(maze_R, goal, cols, rows):
    surf = pygame.Surface((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    surf.fill(BG)
    for (c, r), reward in maze_R.items():
        x    = (c - 1) * TILE_SIZE
        y    = (r - 1) * TILE_SIZE + HUD_H
        rect = pygame.Rect(x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4)
        if (c, r) == goal:
            draw_rounded_rect(surf, GOAL_FILL, rect, 8, GOAL_EDGE, 1)
        elif reward == -20:
            draw_rounded_rect(surf, WALL, rect, 6, WALL_EDGE, 1)
        else:
            draw_rounded_rect(surf, CELL, rect, 6, CELL_EDGE, 1)
    return surf
 
 
# Draws the heatmap maze onto an image.
# Each open cell is coloured by the best Q-value of that cell:
#   red = negative, green = positive, dark = about zero.
# Cells the agent has not updated yet keep the regular maze colour.
# It is rebuilt once per episode because the Q-values change as the agent learns
def build_heatmap_surface(maze_R, maze_Q, goal, cols, rows):
    surf    = pygame.Surface((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    q_min, q_max = compute_q_range(maze_Q, maze_R)
    surf.fill(BG)
    for (c, r), reward in maze_R.items():
        x    = (c - 1) * TILE_SIZE
        y    = (r - 1) * TILE_SIZE + HUD_H
        rect = pygame.Rect(x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4)
        if (c, r) == goal:
            draw_rounded_rect(surf, GOAL_FILL, rect, 8, GOAL_EDGE, 1)
        elif reward == -20:
            draw_rounded_rect(surf, WALL, rect, 6, WALL_EDGE, 1)
        else:
            q_list   = maze_Q.get((c, r), [INITIAL_Q] * 4)
            best_q   = best_learned_q(q_list)
            if best_q is None:
                draw_rounded_rect(surf, CELL, rect, 6, CELL_EDGE, 1)
                continue
            col      = heat_color(q_to_t(best_q, q_min, q_max))
            edge_col = lerp_color(col, (255, 255, 255), 0.12)
            draw_rounded_rect(surf, col, rect, 6, edge_col, 1)
            draw_arrow(surf, rect, q_list.index(best_q), col)
    return surf
 
 
# ── Per-frame drawing (only the things that move) ──────────────────────────
 
def draw_frame(screen, static_surf, maze_R, maze_Q, agent_pos, episode_num,
               T, path_history, start, goal, cols, rows,
               font_large, font_small, tick, q_min=0, q_max=1):
 
    # 1. Draw the pre-built maze image (fast, since we don't redraw every cell)
    screen.blit(static_surf, (0, 0))
 
    # 2. Pulsing goal cell, drawn on top of the maze image every frame
    gc, gr   = goal
    gx       = (gc - 1) * TILE_SIZE
    gy       = (gr - 1) * TILE_SIZE + HUD_H
    g_rect   = pygame.Rect(gx + 2, gy + 2, TILE_SIZE - 4, TILE_SIZE - 4)
    pulse    = 0.5 + 0.5 * math.sin(tick * 0.08)
    glow_col = lerp_color(GOAL_FILL, GOAL_GLOW, pulse)
    draw_rounded_rect(screen, glow_col, g_rect, 8, GOAL_EDGE, 1)
    font_small_g = font_small.render("G", True, (10, 60, 30))
    screen.blit(font_small_g, font_small_g.get_rect(center=g_rect.center))
 
    # 3. Start marker (orange dot)
    sx = int((start[0] - 0.5) * TILE_SIZE)
    sy = int((start[1] - 0.5) * TILE_SIZE) + HUD_H
    pygame.draw.circle(screen, START_FILL, (sx, sy), TILE_SIZE // 8)
 
    # 4. Path trail: only the last PATH_TAIL_LEN steps, fading from dark (old) to bright (new)
    if len(path_history) > 1:
        visible = path_history[-PATH_TAIL_LEN:]
        n = len(visible)
        for i in range(1, n):
            t     = i / n
            color = lerp_color(PATH_TAIL, PATH_HEAD, t)
            p1    = (int((visible[i-1][0] - 0.5) * TILE_SIZE),
                     int((visible[i-1][1] - 0.5) * TILE_SIZE) + HUD_H)
            p2    = (int((visible[i][0]   - 0.5) * TILE_SIZE),
                     int((visible[i][1]   - 0.5) * TILE_SIZE) + HUD_H)
            pygame.draw.line(screen, color, p1, p2, max(1, int(3 * t)))
 
    # 5. The agent (blue dot with a ring around it)
    if agent_pos:
        cx = int((agent_pos[0] - 0.5) * TILE_SIZE)
        cy = int((agent_pos[1] - 0.5) * TILE_SIZE) + HUD_H
        pygame.draw.circle(screen, AGENT_GLOW, (cx, cy), TILE_SIZE // 3, 1)
        pygame.draw.circle(screen, AGENT_FILL, (cx, cy), TILE_SIZE // 6)
 
    # 6. HUD (top info bar): episode counter, temperature bar, move counter
    hud_surf = pygame.Surface((cols * TILE_SIZE, HUD_H))
    hud_surf.fill(HUD_BG)
    screen.blit(hud_surf, (0, 0))
    pygame.draw.line(screen, HUD_ACCENT, (0, HUD_H), (cols * TILE_SIZE, HUD_H), 1)
 
    screen.blit(font_small.render("EPISODE", True, HUD_DIM), (12, 5))
    screen.blit(font_large.render(f"{episode_num} / {TOTAL_EPISODES}", True, HUD_TEXT), (12, 18))
 
    # Temperature bar: green = low T (exploiting), red = high T (exploring)
    bar_x, bar_y, bar_w, bar_h = cols * TILE_SIZE // 3, 10, 120, 6
    pygame.draw.rect(screen, WALL_EDGE, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
    fill_w    = int(bar_w * min(T, 1.0))
    bar_color = lerp_color((60, 200, 80), (255, 80, 80), min(T, 1.0))
    pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=3)
    screen.blit(font_small.render(f"τ = {T:.3f}", True, HUD_DIM), (bar_x, 22))
 
    screen.blit(font_small.render("MOVES", True, HUD_DIM), (cols * TILE_SIZE - 80, 5))
    screen.blit(font_large.render(str(len(path_history)), True, HUD_TEXT), (cols * TILE_SIZE - 80, 18))
 
    # 7. Heatmap legend (bottom-left): colour scale from lowest to highest Q-value,
    #    with a small "0" marking where red turns to green
    leg_x = 8
    leg_y = rows * TILE_SIZE + HUD_H - 20
    leg_w = 140
    leg_h = 8
    for i in range(leg_w):
        q   = q_min + (q_max - q_min) * i / leg_w
        col = heat_color(q_to_t(q, q_min, q_max))
        pygame.draw.line(screen, col,
                         (leg_x + i, leg_y),
                         (leg_x + i, leg_y + leg_h))
    pygame.draw.rect(screen, HUD_DIM, (leg_x, leg_y, leg_w, leg_h), 1)
    lo_lbl = font_small.render(f"{q_min:.0f}", True, HUD_DIM)
    hi_lbl = font_small.render(f"{q_max:.0f}", True, HUD_DIM)
    screen.blit(lo_lbl, (leg_x, leg_y - 13))
    screen.blit(hi_lbl, (leg_x + leg_w - hi_lbl.get_width(), leg_y - 13))
    if q_min < 0 < q_max:
        zero_x   = leg_x + int(leg_w * (0 - q_min) / (q_max - q_min))
        zero_lbl = font_small.render("0", True, HUD_DIM)
        screen.blit(zero_lbl, (zero_x - zero_lbl.get_width() // 2, leg_y - 13))
 
    pygame.display.flip()