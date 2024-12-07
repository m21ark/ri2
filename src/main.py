from math_ops.Math_Ops import Math_Ops as M
from agent.Agent import Agent as Agent_Attacker
from agent.Agent_Penalty import Agent as Agent_Defender
import threading

def attacker_loop():
  while True:
    attacker.think_and_send()
    attacker.scom.receive()
    
def defender_loop():
    while True:
        defender.think_and_send()
        defender.scom.receive()
        
        
def main():
    global attacker, defender
    # Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw, Wait for Server, is magmaFatProxy
    defender = Agent_Defender("localhost", 3100, 3200, 1, "Defender", False, False)
    attacker = Agent_Attacker("localhost", 3100, 3200, 2, "Attacker", False, False)

    # create a thread for each agent
    attacker_thread = threading.Thread(target=attacker_loop)
    defender_thread = threading.Thread(target=defender_loop)
    
    # Setup start scene
    attacker.scom.unofficial_set_play_mode("PlayOn")
    defender.scom.unofficial_set_game_time(0)
    defender.scom.unofficial_move_ball((-8,0,0), (0,0,0))
    defender.scom.unofficial_beam((-14,0,defender.world.robot.beam_height), 0)
    attacker.scom.unofficial_beam((6,0,attacker.world.robot.beam_height), 0)
    
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