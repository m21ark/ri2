from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import numpy as np
import random

class MyAgentDefender(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, enable_log=False, enable_draw=False, wait_for_server=True, is_fat_proxy=False) -> None:
        super().__init__(host, agent_port, monitor_port, 1, 0, "Defender", enable_log, enable_draw, False, wait_for_server, None)
        self.enable_draw = enable_draw
        self.reset()
        
    def reset(self):
        self.state = "init"
        self.kick_dir = 0
        self.reset_kick = True
        self.state_counter = 0

    def think_and_send(self):
        
        # =============== ALIASES ===============
        
        w = self.world
        r = self.world.robot 
        behavior = self.behavior
        my_head_pos_2d = r.loc_head_position[:2]
        
        # =============== DATA ===============
        
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = round(ball_2d[1] - my_head_pos_2d[1],2)
        ball_dist = round(np.linalg.norm(ball_vec),2)
        ball_abs_vel = w.get_ball_abs_vel(12)[:2]
        ball_speed = round(np.linalg.norm(ball_abs_vel),1)
        
        # =============== BEHAVIOR ===============

        # --------------- init state ---------------
        if self.state == "init":
            self.state_counter+=1
            
            # wait for a few iterations before starting
            if self.state_counter > 5:
                self.state = "normal"
                self.state_counter = 0
                
         # --------------- wait for ball shot ---------------
        elif self.state == "normal":
            if ball_speed > 3 and ball_dist < 3: # and abs(ball_dir) > 0.1:
                self.state = "dive"
            
            # adjust current position to be in front of the ball
            else:
                y_coordinate = np.clip(ball_2d[1], -1.1, 1.1)
                behavior.execute("Walk", (-14, y_coordinate), True, 0, True, None)
        
        # --------------- dive to defend ---------------
        elif self.state == "dive":
            dive_dir = "Dive_Left" if ball_dir > 0 else "Dive_Right"
            
            if behavior.execute(dive_dir):
                self.state = "after_dive" # dive is done
                
        # --------------- after dive ---------------
        elif self.state == "after_dive":
            self.state = "get_up" # get up after dive
  
        # --------------- get up ---------------
        elif self.state == "get_up" or behavior.is_ready("Get_Up"):
            
            if behavior.execute("Get_Up"):
                self.state = "normal" # get up is done

        # pubish to world
        self.radio.broadcast()
        self.scom.commit_and_send( r.get_command() )
