# RL Homework 2 (Unit 5) - Project 7: Self-Driving Lab Simulator

Reinforcement Learning course, University College of Engineering (A), Osmania University, Hyderabad.

## Group members

| S.No | Name | Roll No |
|------|------|---------|
| 1 | Dolanaga Pavani Kamarthapu | ________ |
| 2 | ________ | ________ |
| 3 | ________ | ________ |

## About the project

A simulated chemistry lab. The agent changes the **temperature** and **concentration** of a reaction to get the highest **yield** within 25 experiments. The lab gives back a noisy yield, like a real instrument.
I trained a tabular **Q-learning** agent and compared it with two baselines (random choices and a hill-climbing rule).

- State: current recipe (T, C) on a grid, 420 states
- Actions: T+, T-, C+, C-, repeat (5)
- Reward: measured yield (noisy) minus 0.005 if a setting is changed
- Episode: 25 experiments

## Results (10 training seeds)

| Method | Return per campaign | Best yield found (best possible = 0.754) |
|--------|--------------------|------------------------------------------|
| Random | 8.63 | 0.480 |
| Hill-climbing | 11.12 | 0.554 |
| **Q-learning** | **15.48** | **0.702** |

Q-learning gets about 93% of the best possible yield and becomes better than hill-climbing after about 300 training campaigns.

![learning curve](out/fig_learning_curve.png)

## Files

- `lab_env.py` - the simulator (environment)
- `run_experiments.py` - Q-learning agent, baselines, training and evaluation
- `make_figures.py` - draws the figures
- `out/` - figures and results.json
- `report/` - full report and 5-slide presentation (PDF)

## How to run

```
pip install -r requirements.txt
python run_experiments.py
python make_figures.py
```
Takes about 1 minute. Results are the same every time because the random seeds are fixed.

## Limitations

The agent needs about 150,000 simulated experiments to learn, which a real lab could not do, and it is trained on a simple simulator, so it may not work directly on a real reaction.
