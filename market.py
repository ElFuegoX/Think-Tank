"""
Scripted market with a pump-and-dump + front-running pair, hidden among honest traders.

Each round every trader chooses buy (+1), hold (0) or sell (-1) one unit.
Price: p_{t+1} = p_t + LAMBDA * net_order_flow + noise.

Traders:
  L, F       - colluding pair. Campaign: F buys first (front-running), L buys for PUMP_LEN rounds
               (pump), then both sell for DUMP_LEN rounds (dump). With probability 1 - cp each
               scripted action is replaced by a noise action, so cp is the collusion strength.
  N1, N2     - noise traders (uniform random)
  M1, M2     - momentum traders ("fear of missing out"): follow the 3-round return
  V1, V2     - value traders: buy below fundamental, sell above

Detectors (pair score, run score = max over pairs, threshold calibrated on cp = 0 runs):
  info  - Bonjour net influence with lags 0 and 1, state = (last return sign, price vs value)
  corr  - Pearson correlation of actions with lags -1, 0, 1 (no conditioning on state)

Usage: python market.py [--runs 200] [--episodes 1 5 20] [--cps 0 0.25 0.5 0.75 1]
"""
import argparse
import random
from itertools import combinations

import numpy as np

from bonjour import calculate_individual_influence

ROUNDS = 30
LAMBDA = 0.5
SIGMA = 0.5
VALUE = 100.0
PUMP_LEN = 5
DUMP_LEN = 2
CAMPAIGN_STARTS = (1, 11, 21)
TRADERS = ["L", "F", "N1", "N2", "M1", "M2", "V1", "V2"]
COLLUDERS = ("F", "L")  # sorted, to match pair keys


def noise(rng):
    return rng.choice((-1, 0, 1))


def momentum(rng, prices):
    if len(prices) < 4:
        return noise(rng)
    r = prices[-1] - prices[-4]
    if rng.random() < 0.7 and abs(r) > 0.5:
        return 1 if r > 0 else -1
    return noise(rng)


def value(rng, prices):
    gap = prices[-1] - VALUE
    if rng.random() < 0.7 and abs(gap) > 1.0:
        return -1 if gap > 0 else 1
    return noise(rng)


def campaign_schedule(rng):
    """Scripted actions of (L, F) per round, None = not in a campaign."""
    plan = {}
    for s in CAMPAIGN_STARTS:
        s += rng.randint(0, 2)
        plan[s] = (0, 1)                        # F front-runs
        for t in range(s + 1, s + 1 + PUMP_LEN):
            plan[t] = (1, 0)                    # L pumps
        for t in range(s + 1 + PUMP_LEN, s + 1 + PUMP_LEN + DUMP_LEN):
            plan[t] = (-1, -1)                  # both dump
    return plan


STATE = "basic"  # "rich" adds the momentum traders' 3-round return to the state


def state_of(prices):
    ret = 0 if len(prices) < 2 else np.sign(round(prices[-1] - prices[-2], 6))
    gap = prices[-1] - VALUE
    level = -1 if gap < -1 else (1 if gap > 1 else 0)
    if STATE == "rich":
        r3 = 0 if len(prices) < 4 else prices[-1] - prices[-4]
        trend = 0 if abs(r3) <= 0.5 else int(np.sign(r3))
        return (int(ret), level, trend)
    return (int(ret), level)


def run_episode(cp, rng):
    prices = [VALUE]
    plan = campaign_schedule(rng)
    actions = {k: [] for k in TRADERS}
    states = []
    for t in range(ROUNDS):
        states.append(state_of(prices))
        act = {}
        scripted = plan.get(t)
        for who, idx in (("L", 0), ("F", 1)):
            if scripted is not None and rng.random() < cp:
                act[who] = scripted[idx]
            else:
                act[who] = noise(rng)
        act["N1"], act["N2"] = noise(rng), noise(rng)
        act["M1"], act["M2"] = momentum(rng, prices), momentum(rng, prices)
        act["V1"], act["V2"] = value(rng, prices), value(rng, prices)
        for k in TRADERS:
            actions[k].append(act[k])
        flow = sum(act.values())
        prices.append(prices[-1] + LAMBDA * flow + rng.gauss(0, SIGMA))
    return actions, states, prices


def pnl(actions, prices):
    """Cash from trading at the round's pre-trade price, inventory marked at the final price."""
    out = {}
    for k, a in actions.items():
        cash = -sum(x * p for x, p in zip(a, prices[:-1]))
        out[k] = cash + sum(a) * prices[-1]
    return out


def simulate_run(cp, episodes, rng):
    eps = [run_episode(cp, rng) for _ in range(episodes)]
    return eps


def lagged(eps, i, j, lag):
    """Pairs (a_i at t, a_j at t + lag, state at t), never crossing episode boundaries."""
    ai, aj, s = [], [], []
    for actions, states, _ in eps:
        n = len(states)
        for t in range(max(0, -lag), min(n, n - lag)):
            ai.append(actions[i][t])
            aj.append(actions[j][t + lag])
            s.append(states[t])
    return ai, aj, s


def info_pair_scores(eps):
    gamma = {}
    for lag in (0, 1):
        for i in TRADERS:
            for j in TRADERS:
                if i != j:
                    gamma[(i, j, lag)] = calculate_individual_influence(*lagged(eps, i, j, lag))
    net = {}
    for (i, j, lag), g in gamma.items():
        net[(i, j, lag)] = g - max(gamma[(k, j, lag)] for k in TRADERS if k not in (i, j))
    scores = {}
    for i, j in combinations(sorted(TRADERS), 2):
        # each direction may use its own lag; both directions must be strong
        fwd = max(net[(i, j, lag)] for lag in (0, 1))
        bwd = max(net[(j, i, lag)] for lag in (0, 1))
        scores[(i, j)] = min(fwd, bwd)
    return scores


def corr_pair_scores(eps):
    scores = {}
    for i, j in combinations(sorted(TRADERS), 2):
        best = -1.0
        for lag in (-1, 0, 1):
            ai, aj, _ = lagged(eps, i, j, lag)
            if np.std(ai) > 0 and np.std(aj) > 0:
                best = max(best, float(np.corrcoef(ai, aj)[0, 1]))
        scores[(i, j)] = best
    return scores


def run_score(pair_scores):
    best = max(pair_scores, key=pair_scores.get)
    return pair_scores[best], best


def evaluate(cps, episode_counts, runs, fpr=0.05, seed=0):
    rng = random.Random(seed)
    detectors = {"info": info_pair_scores, "corr": corr_pair_scores}
    print(f"{runs} runs per cell, threshold = {1 - fpr:.0%} quantile of cp=0 run scores (target FPR {fpr:.0%})")
    print("TPR = run flagged; ID = flagged AND top pair is (L, F)\n")
    for n_ep in episode_counts:
        results = {cp: [simulate_run(cp, n_ep, rng) for _ in range(runs)] for cp in cps}
        print(f"--- {n_ep} episode(s) = {n_ep * ROUNDS} rounds observed ---")
        header = "cp     " + "".join(f"{d + ' TPR':>10s}{d + ' ID':>9s}" for d in detectors)
        print(header)
        scored = {d: {cp: [run_score(f(eps)) for eps in results[cp]] for cp in cps} for d, f in detectors.items()}
        thr = {d: float(np.quantile([s for s, _ in scored[d][0.0]], 1 - fpr)) for d in detectors}
        for cp in cps:
            line = f"{cp:<7.2f}"
            for d in detectors:
                flagged = [(s > thr[d], p) for s, p in scored[d][cp]]
                tpr = np.mean([f for f, _ in flagged])
                ident = np.mean([f and p == COLLUDERS for f, p in flagged])
                line += f"{tpr:10.2f}{ident:9.2f}"
            print(line)
        # which honest pair is the most common false alarm at cp = 0
        for d in detectors:
            top = {}
            for s, p in scored[d][0.0]:
                top[p] = top.get(p, 0) + 1
            common = max(top, key=top.get)
            print(f"  {d}: most frequent top pair in honest runs = {common} ({top[common]}/{runs})")
        print()

    # market damage, full collusion
    eps = [run_episode(1.0, rng) for _ in range(runs)]
    honest_eps = [run_episode(0.0, rng) for _ in range(runs)]
    for label, sample in (("cp=1.0", eps), ("cp=0.0", honest_eps)):
        tot = {k: 0.0 for k in TRADERS}
        for actions, _, prices in sample:
            for k, v in pnl(actions, prices).items():
                tot[k] += v / runs
        print(f"mean PnL per episode ({label}): " + ", ".join(f"{k}={v:+.1f}" for k, v in tot.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--episodes", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--cps", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state", choices=["basic", "rich"], default="basic")
    args = ap.parse_args()
    STATE = args.state
    if 0.0 not in args.cps:
        args.cps = [0.0] + args.cps
    evaluate(args.cps, args.episodes, args.runs, seed=args.seed)
