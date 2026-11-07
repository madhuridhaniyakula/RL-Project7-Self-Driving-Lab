import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lab_env as L

OUT = "out"
res = json.load(open(f"{OUT}/results.json"))
curves = np.load(f"{OUT}/curves.npy")      # seeds x pts x (ep, ret, best, final)
Q = np.load(f"{OUT}/Q_seed0.npy")
INK, GRID = "#1f2933", "#d9dee3"
C_Q, C_H, C_R = "#0b6e99", "#d9822b", "#7b8794"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

# 1 ---- landscape -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=200)
im = ax.imshow(L.YIELD_GRID, origin="lower", aspect="auto", cmap="viridis",
               extent=[L.C_VALUES[0] - .05, L.C_VALUES[-1] + .05, L.T_VALUES[0] - 2.5, L.T_VALUES[-1] + 2.5])
bi, bj = L.BEST_IDX
ax.plot(L.C_VALUES[bj], L.T_VALUES[bi], marker="*", ms=16, mfc="white", mec=INK)
ax.annotate(f"optimum  {L.T_VALUES[bi]:.0f} °C, {L.C_VALUES[bj]:.1f} mol/L\nyield {L.BEST_YIELD:.2f}",
            (L.C_VALUES[bj], L.T_VALUES[bi]), xytext=(0.95, 105), color="white", fontsize=9,
            arrowprops=dict(arrowstyle="-", color="white"))
ax.set_xlabel("Concentration C (mol/L)"); ax.set_ylabel("Temperature T (°C)")
cb = fig.colorbar(im, ax=ax); cb.set_label("True yield (0–1)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_landscape.png"); plt.close(fig)

# 2 ---- learning curves ------------------------------------------------------
ep = curves[0, :, 0]
fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=200)
m, s = curves[:, :, 1].mean(0), curves[:, :, 1].std(0)
ax.fill_between(ep, m - s, m + s, color=C_Q, alpha=.18, lw=0)
ax.plot(ep, m, color=C_Q, lw=2, label="Q-learning (mean ± sd, 10 seeds)")
ax.axhline(res["hill"]["return"], color=C_H, ls="--", lw=1.8, label=f"Hill-climbing heuristic ({res['hill']['return']:.1f})")
ax.axhline(res["random"]["return"], color=C_R, ls=":", lw=1.8, label=f"Random selection ({res['random']['return']:.1f})")
ax.set_xlabel("Training episodes (1 episode = 1 campaign of 25 experiments)")
ax.set_ylabel("Return per campaign\n(greedy evaluation)")
ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_learning_curve.png"); plt.close(fig)

# 3 ---- best-yield-found comparison -----------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2), dpi=200)
names = ["Random", "Hill-climb", "Q-learning"]
cols = [C_R, C_H, C_Q]
for ax, key, title in zip(axs, ["best_found", "final_yield"],
                          ["Best yield found in campaign", "Yield at end of campaign"]):
    vals = [res["random"][key], res["hill"][key], res["qlearning"][key]]
    ax.bar(names, vals, color=cols, width=.62)
    ax.axhline(L.BEST_YIELD, color=INK, ls="--", lw=1)
    ax.text(2.45, L.BEST_YIELD + .012, "true optimum 0.75", ha="right", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, .9); ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
axs[0].set_ylabel("True yield (0–1)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bars.png"); plt.close(fig)

# 4 ---- learned policy -------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=200)
ax.imshow(L.YIELD_GRID, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1.2,
          extent=[-.5, L.N_C - .5, -.5, L.N_T - .5])
dx = {L.T_UP: (0, .38), L.T_DOWN: (0, -.38), L.C_UP: (.38, 0), L.C_DOWN: (-.38, 0)}
for it in range(L.N_T):
    for ic in range(L.N_C):
        a = int(np.argmax(Q[L.SelfDrivingLabEnv.encode(it, ic)]))
        if a == L.REPEAT:
            ax.plot(ic, it, "o", color="#c0392b", ms=4)
        else:
            x, y = dx[a]
            ax.arrow(ic - x / 2, it - y / 2, x, y, head_width=.2, head_length=.15, color=C_Q, lw=1)
ax.plot(L.BEST_IDX[1], L.BEST_IDX[0], marker="*", ms=14, mfc="gold", mec=INK)
ax.set_xticks(range(0, L.N_C, 3)); ax.set_xticklabels([f"{L.C_VALUES[i]:.1f}" for i in range(0, L.N_C, 3)])
ax.set_yticks(range(0, L.N_T, 4)); ax.set_yticklabels([f"{L.T_VALUES[i]:.0f}" for i in range(0, L.N_T, 4)])
ax.set_xlabel("Concentration C (mol/L)"); ax.set_ylabel("Temperature T (°C)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_policy.png"); plt.close(fig)
print("figures done")
