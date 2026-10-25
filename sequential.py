"""
Sequential (streaming) detection on 4-player RPS: can we raise the alarm early, at a controlled false-alarm rate?

The detector looks at the game every STRIDE rounds. Looking many times inflates false alarms (an honest game
only needs to cross a per-look threshold once), so the alarm rule is calibrated on the *whole* sequence:
  1. at each look t, a score's empirical p-value is its rank among honest calibration runs at the same t;
  2. a run raises the alarm at the first look whose p-value is below c;
  3. c is the 5% quantile of the smallest p-value seen by each honest calibration run,
     so 5% of honest games ever raise an alarm (checked on fresh honest test runs).

Detectors (from rps4.py): info_perm (bias-corrected net influence) and corr (Cramér's V).

Usage: python sequential.py [--horizon 1000] [--stride 100] [--calib 200] [--runs 100]
"""
import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from rps4 import COLLUDERS, DETECTORS, fmt, simulate4


def trajectory(run, looks, detector):
    """(run score, top pair) at each look, using only the rounds seen so far."""
    actions, states = run
    score_fn = DETECTORS[detector]
    out = []
    for t in looks:
        prefix = {p: a[:t] for p, a in actions.items()}
        s = score_fn(prefix, states[:t])
        best = max(s, key=s.get)
        out.append((s[best], best))
    return out


def p_values(traj, calib_scores):
    """Empirical p-value at each look: share of honest calibration scores at that look >= this score."""
    return [(1 + np.sum(calib_scores[k] >= s)) / (1 + len(calib_scores[k])) for k, (s, _) in enumerate(traj)]


def first_alarm(pvals, c):
    for k, p in enumerate(pvals):
        if p < c:
            return k
    return None


def evaluate(variant, horizon, stride, n_calib, runs, cps, rng, names, pool):
    looks = list(range(stride, horizon + 1, stride))
    print(f"\n--- variant={variant}, horizon {horizon} rounds, a look every {stride} rounds ({len(looks)} looks) ---")
    sims = {"calib": [simulate4(0.0, horizon, variant, rng) for _ in range(n_calib)],
            "honest": [simulate4(0.0, horizon, variant, rng) for _ in range(runs)]}
    for cp in cps:
        sims[cp] = [simulate4(cp, horizon, variant, rng) for _ in range(runs)]

    for d in names:
        work = partial(trajectory, looks=looks, detector=d)
        trajs = {k: list(pool.map(work, v, chunksize=4)) for k, v in sims.items()}
        calib_scores = [np.array([tr[k][0] for tr in trajs["calib"]]) for k in range(len(looks))]
        calib_min_p = [min(p_values(tr, calib_scores)) for tr in trajs["calib"]]
        c = float(np.quantile(calib_min_p, 0.05))
        # naive rule for comparison: a fixed 5% threshold at every look
        naive = sum(any(p < 0.05 for p in p_values(tr, calib_scores)) for tr in trajs["honest"])

        print(f"[{d}] false alarms on fresh honest runs: calibrated rule "
              f"{fmt(sum(first_alarm(p_values(tr, calib_scores), c) is not None for tr in trajs['honest']), runs)}"
              f" | naive 5%-per-look rule {fmt(naive, runs)}")
        print(f"{'cp':6s}{'alarm by end':>20s}{'right pair at alarm':>24s}{'median alarm round':>20s}"
              + "".join(f"{'right pair by ' + str(t):>22s}" for t in (looks[len(looks) // 4], looks[len(looks) // 2])))
        for cp in cps:
            alarms = []
            for tr in trajs[cp]:
                k = first_alarm(p_values(tr, calib_scores), c)
                alarms.append((k, k is not None and tr[k][1] == COLLUDERS))
            n_alarm = sum(k is not None for k, _ in alarms)
            n_right = sum(r for _, r in alarms)
            rounds = [looks[k] for k, _ in alarms if k is not None]
            med = f"{int(np.median(rounds))}" if rounds else "-"
            by = []
            for t in (looks[len(looks) // 4], looks[len(looks) // 2]):
                by.append(sum(r and looks[k] <= t for k, r in alarms))
            print(f"{cp:<6.2f}{fmt(n_alarm, runs):>20s}{fmt(n_right, runs):>24s}{med:>20s}"
                  + "".join(f"{fmt(b, runs):>22s}" for b in by))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--cps", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    ap.add_argument("--variants", nargs="+", default=["independent", "herding"])
    ap.add_argument("--detectors", nargs="+", default=["info_perm", "corr"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("Alarm rule calibrated so that 5% of honest games ever alarm. 'right pair' = alarm raised AND top pair is (A, B).")
    print("Brackets: 95% Wilson confidence interval.")
    rng = random.Random(args.seed)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for variant in args.variants:
            evaluate(variant, args.horizon, args.stride, args.calib, args.runs, args.cps, rng, args.detectors, pool)
