from math_ops.Math_Ops import Math_Ops as M
from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Script import Script

script = Script()
a = script.args # Default arguments in config.json

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
# player = Agent("localhost", 3100, 3200, 7, 1, "TeamName")
player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)


w = player.world
player.scom.unofficial_beam((-3,0,w.robot.beam_height), 0)
player.scom.unofficial_move_ball((-1,2,0), (0,0,0))

getting_up = False

while True:
    player_2d = w.robot.loc_head_position[:2]
    ball_2d = w.ball_abs_pos[:2]
    goal_dir = M.vector_angle((15,0)-player_2d) # Calculates the angle of the vector from the robot's head position to the center of the opponent's goal

    if player.behavior.is_ready("Get_Up") or getting_up:
        getting_up = not player.behavior.execute("Get_Up") # True on completion
    else:
        # Kick the ball if it's in the enemy's field
        if ball_2d[0] > 0:
            player.behavior.execute("Basic_Kick", goal_dir)
        # Robot is not aligned with the ball
        elif M.distance_point_to_segment(player_2d, ball_2d, ball_2d+ M.normalize_vec( ball_2d-(15,0) ) ) > 0.1: # not aligned
            next_pos, next_ori, dist = player.path_manager.get_path_to_ball(x_ori=goal_dir, x_dev=-0.3, torso_ori=goal_dir)
            player.behavior.execute("Walk", next_pos, True, next_ori, True, dist) # the last argument, distance is a value between [0, 0.5] and controls the speed near the target
        else:
            player.behavior.execute("Walk", (15,0), True, goal_dir, True, 0.5)
    
    player.scom.commit_and_send( w.robot.get_command() )
    player.scom.receive()

    # Only use to draw (for debugging purposes)
    w.draw.annotation((*player_2d,0.6), "George", w.draw.Color.white, "my_info", flush=False)
    w.draw.line(player_2d, ball_2d, 3, w.draw.Color.yellow, "my_info", flush=True)



# player.behavior.execute_to_completion("Get_Up") # Only for testing purposes, since it blindly executes the command to the end even if the player is beamed