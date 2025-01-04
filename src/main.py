from math_ops.Math_Ops import Math_Ops as M
import random
from MyAgentAttacker import MyAgentAttacker
from MyAgentDefender import MyAgentDefender
import threading

def attacker_loop():
    while True:
        attacker.think_and_send()
        attacker.scom.receive()
    
def defender_loop():
    while True:
        defender.think_and_send()
        defender.scom.receive()
        
        # if defender.world.M_THEIR_GOAL:
            # reset()
            
def reset():
    # Randomize the start pos
    offsetDepth = random.uniform(-1.5, -0.5)
    ofssetWidth = random.uniform(-1.5, 1.5)
    kick_pos = (-10 + offsetDepth, ofssetWidth, 0)
    
    # set the initial conditions
    attacker.scom.unofficial_set_play_mode("PlayOn")
    defender.scom.unofficial_set_game_time(0)
    defender.scom.unofficial_move_ball(kick_pos, (0,0,0))
    defender.scom.unofficial_beam((-14,0,defender.world.robot.beam_height), 0)
    attacker.scom.unofficial_beam((-kick_pos[0] - 0.5, -kick_pos[1], attacker.world.robot.beam_height), 0)
    
def main():
    global attacker, defender

    # create the agents
    defender = MyAgentDefender("localhost", 3100, 3200)
    attacker = MyAgentAttacker("localhost", 3100, 3200)

    # create a thread for each agent
    attacker_thread = threading.Thread(target=attacker_loop)
    defender_thread = threading.Thread(target=defender_loop)
    
    reset()
    
    # start the threads
    attacker_thread.start()
    defender_thread.start()

    # wait for the threads to finish
    attacker_thread.join()
    defender_thread.join()

    attacker.terminate()
    defender.terminate()

if __name__ == "__main__":
    main()