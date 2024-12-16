from agent.Base_Agent import Base_Agent as Agent
from behaviors.custom.Step.Step import Step
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np

'''
Objective:
Learn how to run forward using step primitive
----------
- class Basic_Run: implements an OpenAI custom gym
- class Train:  implements algorithms to train a new model or test an existing model
'''

class Basic_Run(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:

        self.robot_type = r_type

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0 # to limit episode size

        # self.step_obj : Step = self.player.behavior.get_custom_behavior_object("Step") # Step behavior object

        # State space
        obs_size = 21
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(obs_size,-np.inf,np.float32), high=np.full(obs_size,np.inf,np.float32), dtype=np.float32)

        # Action space
        MAX = np.finfo(np.float32).max
        self.no_of_actions = act_size = 8
        self.action_space = gym.spaces.Box(low=np.full(act_size,-MAX,np.float32), high=np.full(act_size,MAX,np.float32), dtype=np.float32)

        # # memory variables
        # self.initialBallPos =  np.array((0,0,0))
        # self.lastBallPos = self.initialBallPos
        # self.lastPlayerPos =  np.array((0,0, self.player.world.robot.beam_height))
        self.lastAction = np.zeros(3,np.float32) # last action (for observation)
        self.ballHasMoved = False
        self.getting_up = False


    def observe(self, init=False):

        r = self.player.world.robot

        # index       observation              naive normalization
        self.obs[0] = self.step_counter        /100  # simple counter: 0,1,2,3...
        self.obs[1] = r.loc_head_z             *3    # z coordinate (torso)
        self.obs[2] = r.loc_head_z_vel         /2    # z velocity (torso)  
        self.obs[3] = r.imu_torso_orientation  /50   # absolute orientation in deg
        self.obs[4] = r.imu_torso_roll         /15   # absolute torso roll  in deg
        self.obs[5] = r.imu_torso_pitch        /15   # absolute torso pitch in deg
        self.obs[6:9] = r.gyro                 /100  # gyroscope
        self.obs[9:12] = r.acc                 /10   # accelerometer

        
        self.obs[12] =  float(not self.player.behavior.is_ready("Get_Up")) # has fallen
        self.obs[13] =  self.getting_up # is getting up (don't forget to reset when taking action)
        self.obs[14:16] = r.loc_head_position[:2] # Position of the player
        self.obs[16:18] = self.player.world.ball_abs_pos[:2] # Position of the ball

        if init:
            self.obs[18] = 0 # Kicking
            self.obs[19] = 0 # Getting Up
            self.obs[20] = 0 # Walking
        else:
            self.obs[18:21] = self.lastAction

        return self.obs

    def sync(self):
        ''' Run a single simulation step '''
        r = self.player.world.robot
        self.player.scom.commit_and_send( r.get_command() )
        self.player.scom.receive()


    def reset(self):
        '''
        Reset and stabilize the robot
        Note: for some behaviors it would be better to reduce stabilization or add noise
        '''

        self.step_counter = 0
        r = self.player.world.robot
        offset = np.random.uniform(-1, 1)

        # memory variables
        self.initialBallPos = np.array((-8 + offset/2,offset,0))
        self.lastBallPos = self.initialBallPos
        self.lastPlayerPos =  np.array((6 + offset, offset, r.beam_height))
        self.lastAction = np.zeros(3,np.float32)
        self.ballHasMoved = False

        for _ in range(25): 
            self.player.scom.unofficial_beam((self.lastPlayerPos[0],self.lastPlayerPos[1],0.50),0) # beam player continuously (floating above ground)
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()

        # beam player to ground
        self.player.scom.unofficial_beam(self.lastPlayerPos,0) 
        r.joints_target_speed[0] = 0.01 # move head to trigger physics update (rcssserver3d bug when no joint is moving)
        self.sync()

        # stabilize on ground
        for _ in range(7):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.player.scom.unofficial_move_ball(self.lastBallPos, (0,0,0))
            self.sync()


        return self.observe(True)

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()

    def step(self, action):
        
        r = self.player.world.robot

        # index of the highest value in the action array
        custom_behavior = np.argmax(action[:3])

        if custom_behavior == 0:
            self.player.behavior.execute("Basic_Kick", action[3])
            self.getting_up = False
            self.lastAction = np.zeros(3,np.float32)
            self.lastAction[0] = 1

        elif custom_behavior == 1:
            self.player.behavior.execute("Get_Up")
            self.getting_up = True
            self.lastAction = np.zeros(3,np.float32)
            self.lastAction[1] = 1

        elif custom_behavior == 2:
            # orientation needs to be between 0 and 2*pi
            orientation = action[6] % (2*np.pi)
            # clip the distance to [0, 0.5]
            distance = np.clip(action[7], 0, 0.5)
            self.player.behavior.execute("Walk", action[4:6], True, orientation, True, distance)
            self.getting_up = False
            self.lastAction = np.zeros(3,np.float32)
            self.lastAction[2] = 1


        self.sync() # run simulation step
        self.step_counter += 1
         
        reward = self.reward()

        # terminal state: the robot is falling or timeout
        # terminal = r.cheat_abs_pos[2] < 0.3 or self.step_counter > 300
        terminal =  self.step_counter > 300
        return self.observe(), reward, terminal, {}


    def reward(self):
        r = self.player.world.robot
        
        points = 0
        distanceMovedBall = np.linalg.norm(self.lastBallPos[:2] - self.initialBallPos[:2])
        if (self.ballHasMoved or distanceMovedBall > 0.05): # Ball has moved
            self.ballHasMoved = True
            # Ball has moved towards the goal
            prev_dist = np.linalg.norm(self.lastBallPos[:2] - (15,0))
            current_dist = np.linalg.norm(self.player.world.ball_abs_pos - (15,0))
            points = prev_dist - current_dist
            
        else: # Ball has not moved
            prev_dist = np.linalg.norm(self.lastPlayerPos[:2] - self.player.world.ball_abs_pos[:2])
            current_dist = np.linalg.norm(r.loc_head_position[:2] - self.player.world.ball_abs_pos[:2])
            points = prev_dist - current_dist

        # Dont forget to update the positions
        self.lastBallPos = self.player.world.ball_abs_pos
        self.lastPlayerPos = r.loc_head_position
        return points



class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)


    def train(self, args):

        #--------------------------------------- Learning parameters
        n_envs = min(16, os.cpu_count())
        n_steps_per_env = 1024  # RolloutBuffer is of size (n_steps_per_env * n_envs)
        minibatch_size = 64    # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 300000#00
        learning_rate = 3e-4
        folder_name = f'Basic_Run_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        #--------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return Basic_Run( self.ip , self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False )
            return thunk

        servers = Server( self.server_p, self.monitor_p_1000, n_envs+1 ) #include 1 extra server for testing

        env = SubprocVecEnv( [init_env(i) for i in range(n_envs)] )
        eval_env = SubprocVecEnv( [init_env(n_envs)] )

        try:
            if "model_file" in args: # retrain
                model = PPO.load( args["model_file"], env=env, device="cpu", n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate )
            else: # train new model
                model = PPO( "MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate, device="cpu" )

            model_path = self.learn_model( model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env*20, save_freq=n_steps_per_env*200, backup_env_file=__file__ )
        except KeyboardInterrupt:
            sleep(1) # wait for child processes
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return
    
        env.close()
        eval_env.close()
        servers.kill()
        

    def test(self, args):

        # Uses different server and monitor ports
        server = Server( self.server_p-1, self.monitor_p, 1 )
        env = Basic_Run( self.ip, self.server_p-1, self.monitor_p, self.robot_type, True )
        model = PPO.load( args["model_file"], env=env )

        try:
            self.export_model( args["model_file"], args["model_file"]+".pkl", False )  # Export to pkl to create custom behavior
            self.test_model( model, env, log_path=args["folder_dir"], model_path=args["folder_dir"] )
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()


'''
The learning process takes several hours.
A video with the results can be seen at:
https://imgur.com/a/dC2V6Et

Stats:
- Avg. reward:     7.7 
- Avg. ep. length: 5.5s (episode is limited to 6s)
- Max. reward:     9.3  (speed: 1.55m/s)    

State space:
- Composed of all joint positions + torso height
- Stage of the underlying Step behavior

Reward:
- Displacement in the x-axis (it can be negative)
- Note that cheat and visual data is only updated every 3 steps
'''
