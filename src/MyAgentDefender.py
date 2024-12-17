from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import numpy as np
import random


class MyAgentDefender(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, enable_log=False, enable_draw=False, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = 0
        team_name = "Defender"
        super().__init__(host, agent_port, monitor_port, 1, robot_type, team_name, enable_log, enable_draw, False, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = "init"
        self.kick_dir = 0 # kick direction
        self.reset_kick = True # when True, a new random kick direction is generated
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
        ball_abs_vel = w.get_ball_abs_vel(6)[:2]
        ball_speed = round(np.linalg.norm(ball_abs_vel),1)
        
        
        # print(f"State: {self.state}, Ball Vec: {ball_vec}, Ball Dir: {ball_dir}, Ball Dist: {ball_dist}, Ball Speed: {ball_speed}")
        
        # =============== PRINTS ===============
        
        # print(f"Direction: {ball_dir}, Distance: {ball_dist}, Speed: {ball_speed}")
    
        # print(f"State: {self.state}, Play Mode: {PM}, Ball Dir: {ball_dir}, Ball Dist: {ball_dist}, Ball Speed: {ball_speed}")
        
        # Intiates at state 1 for 3 iterations
        # then idles at 7 waiting for the ball to be kicked
        # then does 3 (dive right) for a while and after the dive he stays at 4
        
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
            if ball_speed > 3 and ball_dist < 3.5:
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
                
        # if ball is stopped by the goalie, reset the state        
        if ball_speed <= 0.05 and ball_dist < 0.9:
            print("Ball stopped by goalie")

        # pubish to world
        self.radio.broadcast()
        self.scom.commit_and_send( r.get_command() )
