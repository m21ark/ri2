# GYM

# Input

- If is fallen - 1
- Vector of player to the ball - 2
- Vector of player to goalkeeper - 2
- Offset of goalkeeper to the goal - 1

# Output

- Custom actions:
  - Basic Kick (self,reset, direction, abort=False)
  - Dribble (self, reset, orientation, is_orientation_absolute, speed=1, stop=False)
  - Get_Up (self,reset)
  - Step (self,reset, ts_per_step=7, z_span=0.03, z_max=0.8)
  - Walk (self, reset, target_2d, is_target_absolute, orientation, is_orientation_absolute, distance)

# Iteration 1

## Input

- If is fallen and if getting up - 2
- Position of the player - 2
- Position of the ball - 2
- Player's characteristics - 12
- Previous action - 3

Total: 21

## Output

- Choose action - 3
- Custom actions:
  - Basic Kick - direction (1)
  - Get_Up
  - Walk - target_2d (2), orientation (1), distance (float [0, 0.5])

Total: 8

## Reward

If the ball has not yet moved, give rewards if the player is closer to the ball
If the ball has moved, give rewards by moving closer to the goal
