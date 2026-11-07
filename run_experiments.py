"""
Train a tabular Q-learning agent on the self-driving-lab simulator and compare it
with two baselines (random experiment selection, stochastic hill-climbing).
Writes results.json + figures into ./out
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lab_env as L

OUT = "out"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 10
N_EPISODES = 6000
EVAL_EVERY = 50
EVAL_EPISODES = 100
ALPHA, GAMMA = 0.1, 0.95
EPS_START, EPS_END = 1.0, 0.05
BUDGET = 25


# ------------------------------------------------------------------ agents ---
def run_campaign_policy(env, policy_fn):
    """Run one campaign with policy_fn(state, t) -> action; return stats."""
    s = env.reset()
    ret = 0.0
    last_y = 0.0
    for t in range(env.budget):
        a = policy_fn(s, t)
        s, r, done, info = env.step(a)
        ret += r
        last_y = info["y_true"]
    return ret, env.best_true, last_y


def random_policy(rng):
    return lambda s, t: int(rng.integers(L.N_ACTIONS))


def make_hill_climber(env, rng):
    """Stochastic hill climber (model-free heuristic baseline).
    Try a random move; if the measured yield dropped, spend the next experiment
    undoing it. Every experiment counts against the same budget as the RL agent."""
    inverse = {L.T_UP: L.T_DOWN, L.T_DOWN: L.T_UP, L.C_UP: L.C_DOWN, L.C_DOWN: L.C_UP}

    def run(env_):
        env_.reset()
        ret, ref_y, undo, y_true = 0.0, None, None, 0.0
        for _ in range(env_.budget):
            was_undo = undo is not None
            a = undo if was_undo else int(rng.integers(4))
            undo = None
            _, r, _, info = env_.step(a)
            ret += r
            y = info["y_meas"]
            y_true = info["y_true"]
            if was_undo:
                continue                      # back at the reference point
            if ref_y is not None and y < ref_y - 0.005:
                undo = inverse[a]             # worse -> revert next experiment
            else:
                ref_y = y                     # better or equal -> new reference
        return ret, env_.best_true, y_true
    return run


class QAgent:
    def __init__(self, rng):
        self.Q = np.zeros((L.N_STATES, L.N_ACTIONS))
        self.rng = rng

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(L.N_ACTIONS))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, done):
        target = r + (0.0 if done else GAMMA * self.Q[s2].max())
        self.Q[s, a] += ALPHA * (target - self.Q[s, a])


def evaluate_greedy(env, agent):
    rets, bests, finals = [], [], []
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        ret = 0.0
        for t in range(env.budget):
            s, r, done, info = env.step(agent.act(s, 0.0))
            ret += r
        rets.append(ret); bests.append(env.best_true); finals.append(info["y_true"])
    return np.mean(rets), np.mean(bests), np.mean(finals)


def train(seed):
    rng = np.random.default_rng(seed)
    env = L.SelfDrivingLabEnv(budget=BUDGET, seed=seed + 1000)
    eval_env = L.SelfDrivingLabEnv(budget=BUDGET, seed=seed + 2000)
    agent = QAgent(rng)
    train_returns, curve = [], []
    for ep in range(N_EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / (0.7 * N_EPISODES))
        s = env.reset()
        ret = 0.0
        for t in range(BUDGET):
            a = agent.act(s, eps)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s, ret = s2, ret + r
        train_returns.append(ret)
        if ep % EVAL_EVERY == 0 or ep == N_EPISODES - 1:
            curve.append((ep,) + evaluate_greedy(eval_env, agent))
    return agent, np.array(train_returns), np.array(curve)


def baseline_stats(kind, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    env = L.SelfDrivingLabEnv(budget=BUDGET, seed=seed)
    out = []
    if kind == "random":
        pol = random_policy(rng)
        for _ in range(n):
            out.append(run_campaign_policy(env, pol))
    else:
        run = make_hill_climber(env, rng)
        for _ in range(n):
            out.append(run(env))
    return np.array(out)  # columns: return, best_true, final_true


if __name__ == "__main__":
    print(f"True optimum: T={L.T_VALUES[L.BEST_IDX[0]]} C, C={L.C_VALUES[L.BEST_IDX[1]]} M, yield={L.BEST_YIELD:.3f}")
    agents, curves, trains = [], [], []
    for sd in range(N_SEEDS):
        ag, tr, cv = train(sd)
        agents.append(ag); trains.append(tr); curves.append(cv)
        print(f"seed {sd}: final greedy return {cv[-1,1]:.2f}, best-yield {cv[-1,2]:.3f}, final-yield {cv[-1,3]:.3f}")
    curves = np.array(curves)       # seeds x points x 4 (ep, return, best, final)
    trains = np.array(trains)

    rnd = baseline_stats("random")
    hc = baseline_stats("hill")
    q_final = curves[:, -1, 1:]     # seeds x 3
    results = {
        "optimum": {"T_C": float(L.T_VALUES[L.BEST_IDX[0]]), "C_molL": float(L.C_VALUES[L.BEST_IDX[1]]),
                    "yield": L.BEST_YIELD},
        "config": dict(N_SEEDS=N_SEEDS, N_EPISODES=N_EPISODES, ALPHA=ALPHA, GAMMA=GAMMA,
                       EPS_START=EPS_START, EPS_END=EPS_END, BUDGET=BUDGET, noise_sd=0.03, move_cost=0.005),
        "random": dict(zip(["return", "best_found", "final_yield"], rnd.mean(0).tolist())),
        "random_sd": dict(zip(["return", "best_found", "final_yield"], rnd.std(0).tolist())),
        "hill": dict(zip(["return", "best_found", "final_yield"], hc.mean(0).tolist())),
        "hill_sd": dict(zip(["return", "best_found", "final_yield"], hc.std(0).tolist())),
        "qlearning": dict(zip(["return", "best_found", "final_yield"], q_final.mean(0).tolist())),
        "qlearning_sd_across_seeds": dict(zip(["return", "best_found", "final_yield"], q_final.std(0).tolist())),
        "first_eval_q": dict(zip(["return", "best_found", "final_yield"], curves[:, 0, 1:].mean(0).tolist())),
    }
    # episodes needed for Q-learning to beat hill-climber's mean return
    mean_curve = curves[:, :, 1].mean(0)
    eps_axis = curves[0, :, 0]
    beat = np.flatnonzero(mean_curve > results["hill"]["return"])
    results["episodes_to_beat_hill"] = int(eps_axis[beat[0]]) if len(beat) else None
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))

    # greedy policy path from a fixed corner start, for the report
    ag = agents[0]
    np.save(f"{OUT}/Q_seed0.npy", ag.Q)
    np.save(f"{OUT}/curves.npy", curves)
    np.save(f"{OUT}/trains.npy", trains)
    np.save(f"{OUT}/baseline_random.npy", rnd)
    np.save(f"{OUT}/baseline_hill.npy", hc)
