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

# This class implements an OpenAI custom gym. The main methods are:
# - __init__: Initializes the environment
# - observe: Gathers the current state of the agent and environment to form the observation vector
# - reset: Resets the environment to a stable initial state
# - step: Defines how the environment evolves with each agent action
# - reward: Calculates the reward based on the agents's behavior
class MyPenalty(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        # Initialize the agent
        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 2, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0 # to limit episode size

        # ================ State Space ================ #
        obs_size = 21 # 21-dimensional continuous space
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(obs_size,-np.inf,np.float32), high=np.full(obs_size,np.inf,np.float32), dtype=np.float32)

        # ================ Action Space ================ #
        MAX = np.finfo(np.float32).max
        self.no_of_actions = act_size = 8 # 8-dimensional continuous space
        self.action_space = gym.spaces.Box(low=np.full(act_size,-MAX,np.float32), high=np.full(act_size,MAX,np.float32), dtype=np.float32)

        # ================ Agent Memory ================ #
        self.initialBallPos =  np.array((0,0,0))
        self.lastBallPos = self.initialBallPos
        self.lastPlayerPos =  np.array((0,0, self.player.world.robot.beam_height))
        self.lastAction = np.zeros(3,np.float32) # last action (for observation)
        self.ballHasMoved = False
        self.getting_up = False

    # Gathers the current state of the agent and environment to form the observation vector
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

    # Resets the environment to for a new episode
    def reset(self):

        self.step_counter = 0
        r = self.player.world.robot
        
        # Randomize the start pos
        offsetDepth = np.random.uniform(-1.5, -0.5)
        ofssetWidth = np.random.uniform(-1.5, 1.5)
        kick_pos = np.array((-10 + offsetDepth, ofssetWidth, 0))
        
        # set player position
        self.lastPlayerPos =  np.array((-kick_pos[0] - 0.5, -kick_pos[1], r.beam_height))
        
        for _ in range(26): 
            self.player.scom.unofficial_beam(self.lastPlayerPos,0) 
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()
            
        r.joints_target_speed[0] = 0.01 # move head to trigger physics update
            
        # wipe action memory
        self.lastAction = np.zeros(3,np.float32)
        self.ballHasMoved = False

        # set ball position
        self.initialBallPos = kick_pos
        self.lastBallPos = kick_pos
        
        for _ in range(7):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.player.scom.unofficial_move_ball(kick_pos, (0,0,0))
            self.sync()

        return self.observe(True)

    # Defines how the environment evolves with each action
    def step(self, action):
        
        r = self.player.world.robot

        # Determines the robot's behavior based on the highest value in the action array
        # index of the highest value in the action array
        custom_behavior = np.argmax(action[:3])

        if custom_behavior == 0:
            self.player.behavior.execute("Basic_Kick", action[3]) # action[3] is the kick direction
            self.getting_up = False
            self.lastAction = np.zeros(3,np.float32)
            self.lastAction[0] = 1

        elif custom_behavior == 1:
            self.player.behavior.execute("Get_Up")
            self.getting_up = True
            self.lastAction = np.zeros(3,np.float32)
            self.lastAction[1] = 1

        elif custom_behavior == 2:
            orientation = action[6] % (2*np.pi)
            distance = np.clip(action[7], 0, 0.5)
            self.player.behavior.execute("Walk", action[4:6], True, orientation, True, distance)
            self.getting_up = False
            self.lastAction = np.zeros(3,np.float32)
            self.lastAction[2] = 1

        # Run simulation step and get reward
        self.sync()
        self.step_counter += 1
        reward = self.reward()

        # Check if the episode is over
        
        # terminate episode if the robot has fallen 
        terminate = self.player.behavior.is_ready("Get_Up")
        
        # or the ball has reached the goal
        # terminate = terminate or (self.player.world.ball_abs_pos[0] < -15 and abs(self.player.world.ball_abs_pos[1]) < 2)
    
        # terminate if 5 seconds have passed
        terminate = self.step_counter > 500

        return self.observe(), reward, terminate, {}


    # Calculates the reward based on the robot's behavior
    def reward(self):
        r = self.player.world.robot
        
        points = 0
        distanceMovedBall = np.linalg.norm(self.lastBallPos[:2] - self.initialBallPos[:2])

        if (self.ballHasMoved or distanceMovedBall > 0.05): # Ball has moved
            self.ballHasMoved = True
            # Ball has moved towards the goal
            prev_dist = np.linalg.norm(self.lastBallPos[:2] - (-15,0))
            current_dist = np.linalg.norm(self.player.world.ball_abs_pos[:2] - (-15,0))
            points = prev_dist - current_dist * 10 # 10 points per meter
            
        else: # Ball has not moved. Player should move towards the ball
            prev_dist = np.linalg.norm(self.lastPlayerPos[:2] - self.player.world.ball_abs_pos[:2])
            current_dist = np.linalg.norm(r.loc_head_position[:2] - self.player.world.ball_abs_pos[:2])
            points = prev_dist - current_dist
            
        # if the ball reaches the goal
        if self.player.world.ball_abs_pos[0] < -15 and abs(self.player.world.ball_abs_pos[1]) < 2:
            points = 1000
            print("Goal!")
            
        if self.player.behavior.is_ready("Get_Up"):
            points -= 1000
            # print("Fell!")
            
        if self.player.behavior.is_ready("Basic_Kick"):
            points += 50
            # print("Kicked!")

        # Dont forget to update the positions
        self.lastBallPos = self.player.world.ball_abs_pos
        self.lastPlayerPos = r.loc_head_position
        return points
    
    # ================== Helper Functions ================== #
    def sync(self):
        r = self.player.world.robot
        self.player.scom.commit_and_send( r.get_command() )
        self.player.scom.receive()       
    def render(self, mode='human', close=False):
        return
    def close(self):
        Draw.clear_all()
        self.player.terminate()

# Don't change this class because it implements algorithms
# to train a new model or test an existing model
class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):

        #--------------------------------------- Learning parameters
        n_envs = min(16, os.cpu_count())
        n_steps_per_env = 1024  # RolloutBuffer is of size (n_steps_per_env * n_envs)
        minibatch_size = 64    # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 300000 # 00
        learning_rate = 3e-3
        folder_name = f'Penalty_Kick{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        #--------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return MyPenalty( self.ip , self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False )
            return thunk

        servers = Server( self.server_p, self.monitor_p_1000, n_envs+1 )
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
        env = MyPenalty( self.ip, self.server_p-1, self.monitor_p, self.robot_type, True )
        model = PPO.load( args["model_file"], env=env )

        try:
            self.export_model( args["model_file"], args["model_file"]+".pkl", False )  # Export to pkl to create custom behavior
            self.test_model( model, env, log_path=args["folder_dir"], model_path=args["folder_dir"] )
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()
