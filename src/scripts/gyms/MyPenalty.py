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
        obs_size = 10 # 10-dimensional continuous space
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(obs_size,-np.inf,np.float32), high=np.full(obs_size,np.inf,np.float32), dtype=np.float32)

        # ================ Action Space ================ #
        MAX = np.finfo(np.float32).max
        self.num_actions = 6 # 6-dimensional continuous space  
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1, 0, -np.pi, -3, -3], dtype=np.float32),
            high=np.array([1,  1, 0.5,  np.pi,  3,  3], dtype=np.float32),
            dtype=np.float32
        )

        # ================ Agent Memory ================ #
        self.initialBallPos =  np.array((0,0,0))
        self.lastBallPos = self.initialBallPos
        self.lastPlayerPos =  np.array((0,0, self.player.world.robot.beam_height))
        self.lastAction = 0
        self.ballHasMoved = False
        
        # ================ Start Position ================ #
        
        self.kickPos = [10, 0] # X, Y
        self.goalPos = np.array((15,0)) # goal center position
        self.goalWidth = 2
        self.kickerBackOffset = 0.5
        
        
        self.hasFallenLastIteration = False
        

    # Gathers the current state of the agent and environment to form the observation vector
    def observe(self, init=False):

        r = self.player.world.robot
        
        # vector player to goal
        self.goal_player_vec = self.goalPos - r.cheat_abs_pos[:2]
        
        # ==================== Setting the observation space ==================== 
        self.obs[:8] = np.concatenate((
            [self.step_counter / 100],                      # simple counter: 0,1,2,3...
            [self.lastAction],                              # Kicking or walking (0 or 1)
            r.cheat_abs_pos[:2],                            # Position of the player
            self.player.world.ball_abs_pos[:2],             # Position of the ball
            self.goal_player_vec                            # Vector player pos to goal
        ))
        self.obs[8] = r.imu_torso_orientation / 180.0      # absolute orientation in degrees
    
        return self.obs

    # Resets the environment to for a new episode
    def reset(self):

        self.step_counter = 0
        r = self.player.world.robot
        
        # Randomize the start pos: Negative gets away from the goal
        offsetDepth = 0 # np.random.uniform(0, 1.5)
        ofssetWidth = 0 # np.random.uniform(-1.5, 1.5)
        
        kick_pos = np.array((self.kickPos[0] + offsetDepth, self.kickPos[1] + ofssetWidth, 0))
        self.lastPlayerPos =  np.array((kick_pos[0] - self.kickerBackOffset, kick_pos[1], r.beam_height))
        
        self.player.scom.unofficial_set_play_mode("PlayOn")
        
        # set player position
        for _ in range(25): 
            self.player.scom.unofficial_beam([self.lastPlayerPos[0], self.lastPlayerPos[1], 0.5],0) 
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()
            
        self.player.scom.unofficial_beam(self.lastPlayerPos,0) 
        r.joints_target_speed[0] = 0.01 # move head to trigger physics update
        self.sync()
            
        # wipe action memory
        self.lastAction = 0
        self.ballHasMoved = False
        self.initialBallPos = kick_pos
        self.lastBallPos = kick_pos
        self.hasFallenLastIteration = False
        
        for _ in range(7):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.player.scom.unofficial_move_ball(kick_pos, (0,0,0))
            self.sync()
                    
        return self.observe(True)

    # Defines how the environment evolves with each action
    def step(self, action):
        
        r = self.player.world.robot
        
        # Unpack the actions
        isKickOrwalk = action[0] > 0.5
        kick_direction = action[1]
        walk_distance = action[2]
        walk_orientation = action[3]
        walk_target_pos = [action[4], action[5]]
        
        # Execute the action step
        if isKickOrwalk: # Kick 
            self.player.behavior.execute("Basic_Kick", kick_direction)
        else: # Walk   
            self.player.behavior.execute("Walk", walk_target_pos, True, walk_orientation, True, walk_distance)
        self.lastAction = isKickOrwalk

        # Run simulation step and get reward
        self.sync()
        self.step_counter += 1
        reward = self.reward()

        # =============== Check if the episode should be terminated =============== 
        
        
        # terminate episode if the episode has run for too long
        isOvertime = self.step_counter >= 1000

        dist2Ball = abs(np.linalg.norm(r.cheat_abs_pos[:2] - self.player.world.ball_abs_pos[:2]))    
        
        # print("[Ball Pos]: ",self.player.world.ball_abs_pos[:2])
        # print("[Player Pos]: ",r.cheat_abs_pos[:2])
        # print("[Distance BP]: ",np.linalg.norm(r.cheat_abs_pos[:2] - self.player.world.ball_abs_pos[:2]))
        # print("[Vector]: ", self.goal_player_vec)
        # exit(0)
        
        farFromBall = dist2Ball > 3
        
        if farFromBall and not self.ballHasMoved:
            print(f"Episode terminated early due to distance ({round(dist2Ball,2)}) at step {self.step_counter}")
            
        if self.hasFallenLastIteration and not self.ballHasMoved:
            print(f"Episode terminated early due to fall at step {self.step_counter}")

        if isOvertime:
            print(f"Episode terminated early due to time at step {self.step_counter}")
            
        terminate = self.hasFallenLastIteration or isOvertime

        return self.observe(), reward, terminate, {}


    # Calculates the reward based on the robot's behavior
    def reward(self):
        r = self.player.world.robot
        
        points = 0
        distanceMovedBall = np.linalg.norm(self.lastBallPos[:2] - self.initialBallPos[:2])

        if (self.ballHasMoved or distanceMovedBall > 0.05): # Ball has moved
            self.ballHasMoved = True
            # Ball has moved towards the goal
            prev_dist = abs(np.linalg.norm(self.lastBallPos[:2] - self.goalPos))
            current_dist = abs(np.linalg.norm(self.player.world.ball_abs_pos[:2] - self.goalPos))
            points += prev_dist - current_dist * 10
            
        else: # Ball has not moved. Player should move towards the ball
            prev_dist = abs(np.linalg.norm(self.lastPlayerPos[:2] - self.player.world.ball_abs_pos[:2]))
            current_dist = abs(np.linalg.norm(r.cheat_abs_pos[:2] - self.player.world.ball_abs_pos[:2]))
            if current_dist > 1:
                points -= abs(current_dist) * 10 # penalize for moving away from the ball
            else:
                points += 5
            
        # if the ball reaches the goal
        if self.player.world.ball_abs_pos[0] >= self.goalPos[0] and abs(self.player.world.ball_abs_pos[1]) < self.goalWidth:
            points += 1000
            print("Goal!")
            
        if (self.player.behavior.is_ready("Get_Up") or r.loc_head_z < 0.3) and not self.ballHasMoved:
            points -= 1000 # penalize falling before kicking the ball
            self.hasFallenLastIteration = True
            
        if self.player.behavior.is_ready("Basic_Kick"):
            points += 100
            
        # each step costs a little
        points -= 0.1

        # Dont forget to update the positions
        self.lastBallPos = self.player.world.ball_abs_pos
        self.lastPlayerPos = r.cheat_abs_pos
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
        n_envs = 8 # min(16, os.cpu_count())
        n_steps_per_env = 1024  # RolloutBuffer is of size (n_steps_per_env * n_envs)
        minibatch_size = 64    # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 1000000 # 30000000
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
