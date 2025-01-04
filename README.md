# RI 2 - RL Penalty Kick

## Project

Penalty Shooting Scenario using RL in RoboCup 3D Simulation.

Tasks:

- Train a player agent to do penalty kicks with RL
- Develop a basic heuristic-based goalie to defend the kicks

![Start Position](https://github.com/user-attachments/assets/461399c6-cce9-4544-8750-e4654715330a)

![Spawning Positions](https://github.com/user-attachments/assets/e2c04fdf-1230-4dca-93e4-39fe5f643571)

### Heuristic-based Defender

![State Machine](https://github.com/user-attachments/assets/d9d9d3da-86dd-4cdf-99c9-43e223ee0405)

**Ball Tracking (Default State):** Agent moves sideways to place itself between the ball and the goal

**Diving Defense:** Agent dives sideways to defend based on the ball’s speed and direction

![Dive Defense](https://github.com/user-attachments/assets/4a723a23-84b4-44b1-b1ef-e7d1aa9807a6)

### RL Attacker Agent

TODO

## How to Run

### Script

Run the command to startup all needed services and then close them with CTRL+C:

```
> bash setup.sh
```

### Manual approach

- Open 3 terminals, and in 2 of them run the following commands:

```
> rcssserver3d
> cd src/robotViz/bin && sh roboviz.sh
```

- The last command is the script of what you want to run. A possibility is:

```
> python3 src/Run_One_vs_One.py
```

### Troubleshooting

It seems that **rcssserver3d** sometimes gets bugged, and even if you try stop it, it will still be running, making it so the next time you run the UI, it will just display a black screen. To fix this, you need to kill the process that is still running in the background.

```bash
ps aux | grep rcssserver3d # or sudo netstat -plten
kill -9 <pid>
```

## Group

- João Alves (up202007614)
- Marco André (up202004891)
- Rúben Monteiro (up202006478)
