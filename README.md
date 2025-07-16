# RI 2 - RL Penalty Kick

## Project

Penalty Shooting Scenario using Reinforcement Learning in RoboCup 3D Simulation.

Tasks:

- Train a player agent to do penalty kicks with RL
- Develop a basic heuristic-based goalie to defend the kicks

![Start Position](https://github.com/user-attachments/assets/461399c6-cce9-4544-8750-e4654715330a)

See a demo of this project on Youtube:

[![Youtube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=fjvVxMVGUtA)

Or, alternatively, read our [scientific paper](delivery/sci_paper/paper.pdf) on the project.

## How to Run

### To Test the Models

There are three penalty scripts

In the `src/robotViz/bin` run the code below to open the Simulator UI:

```
> sh roboviz.sh
```

Then, with the UI running, in `src/` run:

```
> python3 run_utils.py
```

In the menu choose which task you want to run (nº 18 - 20) and then select to test with the best model.

For both `MyPenalty_1` and `MyPenalty_2`, the best model is `T1_Best_No_Goalie`. 

For `MyPenalty_3`, the best model is `T3_Best_With_Goalie`.

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
