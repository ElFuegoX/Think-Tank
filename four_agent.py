"""
Four-agent repeated RPS with one seeded colluding pair, plus a three-agent cartel variant.

Scenarios:
  pair   A, B collude: B plays the move A beats with probability cp; C, D play at random.
  decoy  as pair, plus an honest "follower" D that copies C's previous move with probability 0.5
         (a lagged, public-looking dependence that is not collusion).
  cartel A, B, C collude as a triad: C plays (A + B) mod 3 with probability cp. Every pair is
         independent, so only a triple-level measure (synergy) can see it.

Detectors are calibrated on cp = 0 runs to a 5% false-alarm rate; ID = flagged and the top pair is
the colluding pair (for cartel: the top triple is (A, B, C)).

Usage: python four_agent.py [--runs 200] [--rounds 100 300 1000] [--cps 0 0.1 0.2 0.4]
"""
import argparse
import random

from detectors import DETECTORS, run_score, calibrate, wilson, synergy

MOVES = ["R", "P", "S"]
BEATS = {"R": "S", "P": "R", "S": "P"}
PLAYERS = ["A", "B", "C", "D"]


def play(scenario, cp, rounds, rng):
    acts = {p: [] for p in PLAYERS}
    for t in range(rounds):
        a, c = rng.choice(MOVES), rng.choice(MOVES)
        if scenario == "cartel":
            b = rng.choice(MOVES)
            c = MOVES[(MOVES.index(a) + MOVES.index(b)) % 3] if rng.random() < cp else c
        else:
            b = BEATS[a] if rng.random() < cp else rng.choice(MOVES)
        if scenario == "decoy" and t > 0 and rng.random() < 0.5:
            d = acts["C"][-1]
        else:
            d = rng.choice(MOVES)
        for p, m in zip(PLAYERS, (a, b, c, d)):
            acts[p].append(m)
    return acts


def evaluate(scenario, cps, rounds_list, runs, fpr=0.05, seed=0):
    rng = random.Random(seed)
    truth = ("A", "B")
    print(f"\n##### scenario = {scenario}  ({runs} runs per cell, FPR target {fpr:.0%}, ID with 95% Wilson CI)")
    for rounds in rounds_list:
        data = {cp: [play(scenario, cp, rounds, rng) for _ in range(runs)] for cp in cps}
        print(f"--- {rounds} rounds ---")
        if scenario == "cartel":
            dets = {"synergy": synergy, **DETECTORS}
            target = ("A", "B", "C")
        else:
            dets, target = DETECTORS, truth
        scored = {d: {cp: [run_score(f(a)) for a in data[cp]] for cp in cps} for d, f in dets.items()}
        thr = {d: calibrate([s for s, _ in scored[d][0.0]], fpr) for d in dets}
        print("cp     " + "".join(f"{d:>26s}" for d in dets))
        for cp in cps:
            cells = []
            for d in dets:
                flagged = [(s > thr[d], p) for s, p in scored[d][cp]]
                if cp == 0:
                    k = sum(f for f, _ in flagged)
                    label = "FPR"
                else:
                    good = target if d == "synergy" else truth
                    k = sum(f and (p == good or (scenario == "cartel" and set(p) <= set(target))) for f, p in flagged)
                    label = "ID"
                lo, hi = wilson(k, runs)
                cells.append(f"{label} {k / runs:.2f} [{lo:.2f},{hi:.2f}]")
            print(f"{cp:<7.2f}" + "".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--rounds", type=int, nargs="+", default=[100, 300, 1000])
    ap.add_argument("--cps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    ap.add_argument("--scenarios", nargs="+", default=["pair", "decoy", "cartel"])
    args = ap.parse_args()
    if 0.0 not in args.cps:
        args.cps = [0.0] + args.cps
    for sc in args.scenarios:
        evaluate(sc, args.cps, args.rounds, args.runs)
