"""
Self-driving lab simulator (RL Homework 2, Project 7)
Each step the agent changes the temperature or the concentration a little and
the lab gives back a noisy measured yield. Goal: find the (T, C) with the
highest yield in a fixed number of experiments.
"""
import numpy as np

R = 8.314  # J / (mol K)

# ---- discretised recipe grid -------------------------------------------------
T_VALUES = np.arange(20, 121, 5.0)        # temperature in deg C  (21 levels)
C_VALUES = np.round(np.arange(0.1, 2.01, 0.1), 2)  # concentration in mol/L (20 levels)
N_T, N_C = len(T_VALUES), len(C_VALUES)
N_STATES = N_T * N_C

# action ids
T_UP, T_DOWN, C_UP, C_DOWN, REPEAT = range(5)
ACTION_NAMES = ["T+", "T-", "C+", "C-", "repeat"]
N_ACTIONS = 5


def true_yield(T_c, C):
    """True yield (0 to 1) of the reaction A -> P, with a side reaction.
    conversion    : Arrhenius rate, saturates as concentration increases
    selectivity   : side reaction has higher activation energy and is 2nd order,
                    so high T and high C both reduce selectivity
    decomposition : product breaks down above about 90 C
    """
    T = T_c + 273.15
    Tref = 60 + 273.15
    k1 = np.exp(50e3 / R * (1 / Tref - 1 / T))        # main reaction, Ea = 50 kJ/mol
    k2 = 0.15 * np.exp(85e3 / R * (1 / Tref - 1 / T))  # side reaction, Ea = 85 kJ/mol
    conversion = 1.0 - np.exp(-1.6 * k1 * C ** 0.6)
    selectivity = 1.0 / (1.0 + (k2 / k1) * C * 1.2)
    decomposition = 1.0 / (1.0 + np.exp((T_c - 92.0) / 5.0))
    return conversion * selectivity * decomposition


YIELD_GRID = np.array([[true_yield(t, c) for c in C_VALUES] for t in T_VALUES])
BEST_IDX = np.unravel_index(np.argmax(YIELD_GRID), YIELD_GRID.shape)
BEST_YIELD = float(YIELD_GRID[BEST_IDX])


class SelfDrivingLabEnv:
    """Simple gym-style environment (reset / step), only numpy."""

    def __init__(self, budget=25, noise_sd=0.03, move_cost=0.005, seed=None):
        self.budget = budget          # experiments per campaign (episode length)
        self.noise_sd = noise_sd      # measurement noise of the instrument
        self.move_cost = move_cost    # small cost for changing a setting (reagent/time)
        self.rng = np.random.default_rng(seed)

    # state <-> index helpers
    @staticmethod
    def encode(i_t, i_c):
        return i_t * N_C + i_c

    @staticmethod
    def decode(s):
        return divmod(s, N_C)

    def reset(self):
        self.i_t = int(self.rng.integers(N_T))
        self.i_c = int(self.rng.integers(N_C))
        self.t = 0
        self.best_measured = 0.0
        self.best_true = 0.0
        return self.encode(self.i_t, self.i_c)

    def step(self, action):
        moved = action != REPEAT
        if action == T_UP:
            self.i_t = min(self.i_t + 1, N_T - 1)
        elif action == T_DOWN:
            self.i_t = max(self.i_t - 1, 0)
        elif action == C_UP:
            self.i_c = min(self.i_c + 1, N_C - 1)
        elif action == C_DOWN:
            self.i_c = max(self.i_c - 1, 0)
        y_true = YIELD_GRID[self.i_t, self.i_c]
        y_meas = float(np.clip(y_true + self.rng.normal(0, self.noise_sd), 0, 1))
        reward = y_meas - (self.move_cost if moved else 0.0)
        self.t += 1
        self.best_measured = max(self.best_measured, y_meas)
        self.best_true = max(self.best_true, y_true)
        done = self.t >= self.budget
        info = {"y_true": y_true, "y_meas": y_meas}
        return self.encode(self.i_t, self.i_c), reward, done, info
