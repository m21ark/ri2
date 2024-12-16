# GYM


# Input
- If is fallen  - 1
- Vector of player to the ball - 2
- Vector of player to goalkeeper - 2
- Offset of goalkeeper to the goal - 1


# Output
- Custom actions:
    - Basic Kick    (self,reset, direction, abort=False)
    - Dribble       (self, reset, orientation, is_orientation_absolute, speed=1, stop=False)
    - Get_Up        (self,reset)
    - Step          (self,reset, ts_per_step=7, z_span=0.03, z_max=0.8)
    - Walk          (self, reset, target_2d, is_target_absolute, orientation, is_orientation_absolute, distance)



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





# Extras that might help with bugs
### Sections 4, 5 and 10 are probably the most important for us
It seems that the **rcssserver3d** sometimes gets bugged, and even if you stop it, the next time you run it, it will just display the black screen. To fix this, you need to kill the process that is still running in the background.
```bash
ps aux | grep rcssserver3d # or sudo netstat -plten 
kill -9 <pid>
```
The **PlayOn** is important, since otherwise the ball won't move from its initial position, even by script.
The default arguments that can be passed through cmdline of the Script are recorded in the **config.json file**. You can change them.
For some reason I can't **change the team names**. I think this might be decided at startup when the server is started, so if you try to change this after the server is running, it won't work, I think.
The `player.behavior.execute()` command updates the internal array w.robot.joints_target_speed with target values for each joint. And these:
```
player.scom.commit_and_send( w.robot.get_command() )
player.scom.receive()
```
just commit the changes, send them to the server, and wait for a response.
