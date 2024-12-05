from agent.Base_Agent import Base_Agent as Agent
from math_ops.Math_Ops import Math_Ops as M
from scripts.commons.Script import Script

script = Script()
a = script.args


# Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
p1 = Agent(a.i, a.p, a.m, a.u, a.r, a.t)
p2 = Agent(a.i, a.p, a.m, a.u, a.r, "Opponent")
players = [p1,p2]


p1.scom.unofficial_beam((-3,0,p1.world.robot.beam_height), 0)
p2.scom.unofficial_beam((-3,0,p2.world.robot.beam_height), 0)


getting_up = [False]*2


while True:
    for i in range(len(players)):
        p = players[i]
        w = p.world


        player_2d = w.robot.loc_head_position[:2]
        ball_2d = w.ball_abs_pos[:2]
        goal_dir = M.vector_angle( (15,0)-player_2d ) # Goal direction


        if p.behavior.is_ready("Get_Up") or getting_up[i]:
            getting_up[i] = not p.behavior.execute("Get_Up") # True on completion
        else:
            p.behavior.execute("Basic_Kick", goal_dir)
          
        p.scom.commit_and_send( w.robot.get_command() )


    # all players must commit and send before the server updates
    p1.scom.receive()
    p2.scom.receive()


# from agent.Agent import Agent

# # Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw
# script.batch_create(Agent, ((a.i, a.p, a.m, a.u, a.t,        True, True),)) #one player for home team
# script.batch_create(Agent, ((a.i, a.p, a.m, a.u, "Opponent", True, True),)) #one player for away team


# while True:
#     script.batch_execute_agent()
#     script.batch_receive()
