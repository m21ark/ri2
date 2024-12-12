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
        self.state = 0  # 0-Normal, 1-Getting up, 2-Dive Left, 3-Dive Right, 4-Wait
        self.kick_dir = 0 # kick direction
        self.reset_kick = True # when True, a new random kick direction is generated

    def think_and_send(self):
        
        # =============== ALIASES ===============
        
        w = self.world
        r = self.world.robot 
        behavior = self.behavior
        my_head_pos_2d = r.loc_head_position[:2]
        
        # =============== DATA ===============
        
        my_ori = r.imu_torso_orientation
        print(f"Ori: {my_ori:5.3f}")
        
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = M.vector_angle(ball_vec)
        ball_dist = np.linalg.norm(ball_vec)
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
    
        # print(f"State: {self.state}, Play Mode: {PM}, Ball Dir: {ball_dir}, Ball Dist: {ball_dist}, Ball Speed: {ball_speed}")
        
        # Intiates at state 1 for 3 iterations
        # then idles at 7 waiting for the ball to be kicked
        # then does 3 (dive right) for a while and after the dive he stays at 4
        
        # =============== BEHAVIOR ===============

        if w.play_mode != w.M_PLAY_ON:
            
            # print("Initiating")
            
            self.state = 0
            self.reset_kick = True
            behavior.execute("Zero_Bent_Knees") # wait
                
        elif self.state == 2:
            # print("Diving Left")
            self.state = 4 if behavior.execute("Dive_Left") else 2  # change state to wait after skill has finished
            
        elif self.state == 3:
            # print("Diving Right")
            self.state = 4 if behavior.execute("Dive_Right") else 3 # change state to wait after skill has finished
            
        elif self.state == 4:
            # print("Waiting after diving")
            pass
        
        elif self.state == 1 or behavior.is_ready("Get_Up"):
            # print("Getting Up")
            self.state = 0 if behavior.execute("Get_Up") else 1 # return to normal state if get up behavior has finished
            
        elif r.unum == 1:
            # print("Waiting for ball to be shot")
            y_coordinate = np.clip(ball_2d[1], -1.1, 1.1)
            behavior.execute("Walk", (-14,y_coordinate), True, 0, True, None) # Args: target, is_target_abs, ori, is_ori_abs, distance
            if ball_2d[0] < -10: 
                self.state = 2 if ball_2d[1] > 0 else 3 # dive to defend
                
        else:
            print("Unknown state")
                
        # pubish to world
        self.radio.broadcast()
        self.scom.commit_and_send( r.get_command() )