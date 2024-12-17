from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np


class MyAgentAttacker(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, enable_log=False, enable_draw=False, unum = 2, wait_for_server=True) -> None:
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]
        super().__init__(host, agent_port, monitor_port, unum, robot_type, "Attacker", enable_log, enable_draw, True, wait_for_server, None)
        self.enable_draw = enable_draw
        self.state = 0
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_walk = np.zeros(3) 
        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] 


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):

        r = self.world.robot

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)


    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control

    def think_and_send(self):
        
        # =============== ALIASES ===============
                
        w = self.world
        r = self.world.robot  
        behavior = self.behavior
        my_head_pos_2d = r.loc_head_position[:2]
        my_ori = r.imu_torso_orientation
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        
        # =============== DATA ===============
                
        ball_dir = M.vector_angle(ball_vec)
        ball_dist = np.linalg.norm(ball_vec)
        ball_sq_dist = ball_dist * ball_dist # for faster comparisons
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
        goal_dir = M.target_abs_angle(ball_2d,(15.05,0))

        # =============== Preprocessing ===============
        
        slow_ball_pos = w.get_predicted_ball_pos(0.5) # predicted future 2D ball position when ball speed <= 0.5 m/s

        # list of squared distances between teammates (including self) and slow ball (sq distance is set to 1000 in some conditions)
        teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                  else 1000 # force large distance if teammate does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in w.teammates ]

        # list of squared distances between opponents and slow ball (sq distance is set to 1000 in some conditions)
        opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and w.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                  else 1000 # force large distance if opponent does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in w.opponents ]

        min_teammate_ball_sq_dist = min(teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(min_teammate_ball_sq_dist)   # distance between ball and closest teammate
        self.min_opponent_ball_dist = math.sqrt(min(opponents_ball_sq_dist)) # distance between ball and closest opponent

        # =============== BEHAVIOR ===============

        if self.state == 1 or behavior.is_ready("Get_Up"):
            self.state = 0 if behavior.execute("Get_Up") else 1
            
        else: # normal state

            if self.min_opponent_ball_dist + 0.5 < self.min_teammate_ball_dist: # defend if opponent is considerably closer to the ball
                if self.state == 2: # commit to kick while aborting
                    self.state = 0 if self.kick(abort=True) else 2
                else: # move towards ball, but position myself between ball and our goal
                    self.move(slow_ball_pos + M.normalize_vec((-16,0) - slow_ball_pos) * 0.2, is_aggressive=True)
            else:
                self.state = 0 if self.kick(goal_dir, 20, False, False) else 2

        # pubish to world
        self.radio.broadcast()
        self.scom.commit_and_send( r.get_command() )
