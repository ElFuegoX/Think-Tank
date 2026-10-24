"""
4-player RPS with one seeded colluding pair: net influence vs action correlation (Cramér's V).

Players: A, C, D play at random; B colludes with A (plays the move A beats) with probability cp.
Variants:
  independent - honest players are independent (no common cause)
  herding     - honest C and D each, with probability HERD, play the move that beats A's previous move
                (a public signal), so they are correlated without colluding

Detectors (run score = max over the 6 pairs; threshold = 95% quantile of cp = 0 runs, target FPR 5%):
  info - Bonjour net influence, pair score = min(Γ(i;j), Γ(j;i)), state = previous joint move
  corr - Cramér's V between the two players' moves (no conditioning)

Usage: python rps4.py [--runs 200] [--games 300 1000 3000] [--cps 0.1 0.2 0.4]
"""
import argparse
import random
from collections import Counter
from itertools import combinations

import numpy as np

from bonjour import BEATS, action_space, calculate_individual_influence, calculate_net_influence

PLAYERS = ["A", "B", "C", "D"]
COLLUDERS = ("A", "B")
HERD = 0.5
BEATEN_BY = {v: k for k, v in BEATS.items()}  # move -> the move that beats it


def simulate4(cp, games, variant, rng):
    actions = {p: [] for p in PLAYERS}
    prev_a = None
    for _ in range(games):
        a = rng.choice(action_space)
        b = BEATS[a] if rng.random() < cp else rng.choice(action_space)
        moves = {"A": a, "B": b}
        for p in ("C", "D"):
            if variant == "herding" and prev_a is not None and rng.random() < HERD:
                moves[p] = BEATEN_BY[prev_a]
            else:
                moves[p] = rng.choice(action_space)
        for p in PLAYERS:
            actions[p].append(moves[p])
        prev_a = a
    states = [-1] + [tuple(actions[p][t - 1] for p in PLAYERS) for t in range(1, games)]
    return actions, states


def info_scores(actions, states):
    _, net = calculate_net_influence(actions, states)
    return {(i, j): min(net[(i, j)], net[(j, i)]) for i, j in combinations(PLAYERS, 2)}


def gamma_mm(xi, xj, states):
    """Conditional MI with Miller–Madow bias correction of each entropy (bits)."""
    by_state = {}
    for s, a, b in zip(states, xi, xj):
        by_state.setdefault(s, []).append((a, b))
    total, out = len(states), 0.0
    for pairs in by_state.values():
        n = len(pairs)
        ci = Counter(a for a, _ in pairs)
        cj = Counter(b for _, b in pairs)
        cij = Counter(pairs)
        h = lambda c: -sum(v / n * np.log2(v / n) for v in c.values())
        # H_MM = H_plugin + (bins - 1) / (2 n ln 2); applied to H(i) + H(j) - H(i, j)
        corr = ((len(ci) - 1) + (len(cj) - 1) - (len(cij) - 1)) / (2 * n * np.log(2))
        out += n / total * (h(ci) + h(cj) - h(cij) + corr)
    return out


def gamma_perm(xi, xj, states, rng, k=20):
    """Plug-in conditional MI minus its mean under within-state shuffles of j (pair-specific bias)."""
    idx_by_state = {}
    for t, s in enumerate(states):
        idx_by_state.setdefault(s, []).append(t)
    null = []
    for _ in range(k):
        shuffled = list(xj)
        for idx in idx_by_state.values():
            vals = [xj[t] for t in idx]
            rng.shuffle(vals)
            for t, v in zip(idx, vals):
                shuffled[t] = v
        null.append(calculate_individual_influence(xi, shuffled, states))
    return calculate_individual_influence(xi, xj, states) - float(np.mean(null))


def net_from_gamma(gamma):
    net = {}
    for (i, j), g in gamma.items():
        net[(i, j)] = g - max(gamma[(k, j)] for k in PLAYERS if k not in (i, j))
    return {(i, j): min(net[(i, j)], net[(j, i)]) for i, j in combinations(PLAYERS, 2)}


def info_mm_scores(actions, states):
    gamma = {(i, j): gamma_mm(actions[i], actions[j], states) for i in PLAYERS for j in PLAYERS if i != j}
    return net_from_gamma(gamma)


_perm_rng = random.Random(12345)


def info_perm_scores(actions, states):
    # MI is symmetric, so one shuffle-corrected value per unordered pair
    sym = {(i, j): gamma_perm(actions[i], actions[j], states, _perm_rng) for i, j in combinations(PLAYERS, 2)}
    gamma = {}
    for (i, j), g in sym.items():
        gamma[(i, j)] = gamma[(j, i)] = g
    return net_from_gamma(gamma)


def cramers_v(x, y):
    idx = {m: k for k, m in enumerate(action_space)}
    table = np.zeros((3, 3))
    for a, b in zip(x, y):
        table[idx[a], idx[b]] += 1
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = ((table - expected) ** 2 / np.where(expected > 0, expected, 1)).sum()
    return float(np.sqrt(chi2 / (table.sum() * 2)))


def corr_scores(actions, states):
    return {(i, j): cramers_v(actions[i], actions[j]) for i, j in combinations(PLAYERS, 2)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f},{hi:.2f}]"


DETECTORS = {"info": info_scores, "info_mm": info_mm_scores, "info_perm": info_perm_scores, "corr": corr_scores}


def evaluate(variant, games, cps, runs, fpr, rng, names=("info", "corr")):
    detectors = {d: DETECTORS[d] for d in names}
    sims = {cp: [simulate4(cp, games, variant, rng) for _ in range(runs)] for cp in [0.0] + cps}
    scored = {d: {cp: [] for cp in sims} for d in detectors}
    for cp, runs_cp in sims.items():
        for actions, states in runs_cp:
            for d, f in detectors.items():
                s = f(actions, states)
                best = max(s, key=s.get)
                scored[d][cp].append((s[best], best))
    thr = {d: float(np.quantile([s for s, _ in scored[d][0.0]], 1 - fpr)) for d in detectors}

    print(f"\n--- variant={variant}, {games} rounds, {runs} runs per cell ---")
    print(f"{'cp':6s}" + "".join(f"{d + ' TPR':>24s}{d + ' ID':>24s}" for d in detectors))
    for cp in sims:
        line = f"{cp:<6.2f}"
        for d in detectors:
            flagged = [s > thr[d] for s, _ in scored[d][cp]]
            ident = [s > thr[d] and p == COLLUDERS for s, p in scored[d][cp]]
            line += f"{fmt(sum(flagged), runs):>24s}{fmt(sum(ident), runs):>24s}"
        print(line)
    for d in detectors:
        top = {}
        for _, p in scored[d][0.0]:
            top[p] = top.get(p, 0) + 1
        common = max(top, key=top.get)
        print(f"  {d}: most frequent top pair in honest runs = {common} ({top[common]}/{runs}),"
              f" (A, B) top in {top.get(COLLUDERS, 0)}/{runs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--games", type=int, nargs="+", default=[300, 1000, 3000])
    ap.add_argument("--cps", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    ap.add_argument("--variants", nargs="+", default=["independent", "herding"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detectors", nargs="+", default=["info", "corr"], choices=list(DETECTORS))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    print("cp = 0 row: TPR is the false-alarm rate (target 0.05). ID = flagged AND top pair is (A, B).")
    print("Brackets: 95% Wilson confidence interval.")
    for variant in args.variants:
        for games in args.games:
            evaluate(variant, games, args.cps, args.runs, 0.05, rng, args.detectors)
