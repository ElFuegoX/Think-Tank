---
noteId: "96de79b0cf4d11f19b45d9aa8d8e7b58"
tags: []

---

# Bonjour et al. reimplementation: 3-player RPS, assistant collusion

Date: 2026-10-24. Code: `bonjour.py` (detector + demo + fixed-threshold sweep), `perm_sweep.py` (permutation-calibrated sweep).

## Setup
- 3 players A, B, C play repeated rock-paper-scissors. A and C play uniformly at random.
- B is A's assistant: with probability CP (collusion probability) B plays the move A beats; otherwise B plays at random.
- State `s` = joint action of the previous round (27 states).
- Detector (Bonjour et al., Eq. 6): individual influence γ(i;j) = I(a_i; a_j | s); net influence Γ(i;j) = γ(i;j) − max_{k≠i,j} γ(k;j). A pair is flagged when Γ is above the threshold in **both** directions.

## 1. The original script failed on its own example
With CP = 0.4 and 1,000 games, the original `bonjour.py` did not flag the true pair (A, B). Over 20 seeds:

| Collusion rule | n = 1,000 | n = 10,000 |
|---|---|---|
| Original rule | A–B found 1/20 | A–B found 0/20 |
| Fixed rule | A–B found 20/20 | A–B found 20/20 |
| No collusion | no false alarm (20/20) | no false alarm (20/20) |

**Cause:** in the original rule, when A ≠ C, B played the third move, so B's action depended on both A and C. The collusion signal was spread over A–B and B–C, and the net influence cancelled it. More data did not help.

## 2. Fixes in `bonjour.py`
1. **Collusion rule:** B plays `BEATS[A]` with probability CP, so its colluding move depends only on A. Before the fix, CP was also inverted (`random.random() > ratio`).
2. **Seed:** `random.seed(42)` replaces `np.random.seed(42)`, which did not control the `random` module that generates actions. Runs are now reproducible.
3. **N players:** `calculate_net_influence` takes a `{player: actions}` dict and uses the max over all k ≠ i, j, as in the paper (it was hard-coded to 3 players).
4. **Permutation null:** `permutation_null` shuffles each player's actions within each state. This breaks dependence between players but keeps each player's per-state action distribution, giving p-values for every Γ(i;j).
5. **Helpers:** `detect_collusion` (the two-way threshold rule), `simulate` and `sweep`.

Not changed: the reward function gives 1 to everyone when all three moves differ. This looks unusual, but rewards are not used by the detector.

## 3. Demo run (CP = 0.4, 1,000 games, seed 42)

| Pair | Γ | permutation p |
|---|---|---|
| A → B | 0.180 | 0.005 |
| B → A | 0.188 | 0.005 |
| A → C | −0.008 | 0.72 |
| B → C | 0.008 | 0.29 |
| C → A | −0.188 | 1.00 |
| C → B | −0.180 | 1.00 |

Detected: (A, B). Correct.

## 4. Sweep with the paper's fixed threshold (Γ > 0.05 both ways), 50 seeds per cell
Cells give the share of runs that flag **exactly** (A, B). For CP = 0, cells give the false-alarm rate (any pair flagged). Command: `python bonjour.py --sweep`.

| CP | n = 100 | n = 300 | n = 1,000 | n = 3,000 |
|---|---|---|---|---|
| 0.00 (false alarms) | **0.46** | 0.10 | 0.00 | 0.00 |
| 0.05 | 0.10 | 0.08 | 0.00 | 0.00 |
| 0.10 | 0.12 | 0.10 | 0.04 | 0.00 |
| 0.20 | 0.26 | 0.36 | 0.52 | 0.64 |
| 0.40 | 0.60 | 0.98 | 1.00 | 1.00 |

## 5. Sweep with a permutation-calibrated threshold (p < 0.05 both ways), 30 seeds, 100 permutations
Command: `python perm_sweep.py`.

| CP | n = 100 | n = 300 | n = 1,000 |
|---|---|---|---|
| 0.00 (false alarms) | **0.03** | 0.00 | 0.03 |
| 0.10 | 0.00 | 0.07 | 0.17 |
| 0.20 | 0.13 | 0.07 | 0.73 |
| 0.40 | 0.27 | 0.97 | 1.00 |

## 6. Findings
1. **The method works when collusion is strong and the data is sufficient.** At CP = 0.4, it finds the right pair in every run from about 300 games on.
2. **A fixed threshold does not control false alarms with little data.** At 100 games, 46% of honest runs are flagged. The plug-in estimate of γ is biased upward (about 0.08 bits for an independent pair at 1,000 games, about 0.008 at 10,000). Net influence cancels most of this bias, but not enough with very few samples.
3. **A fixed threshold sets a hard floor on detectable collusion.** The true signal is I(A;B) = log2 3 − H(q, (1−q)/2, (1−q)/2) with q = CP + (1−CP)/3. That gives about 0.014 bits at CP = 0.1, 0.055 at CP = 0.2 and 0.21 at CP = 0.4. Below about CP = 0.19, Γ converges *under* 0.05, so more data makes detection **worse** (CP = 0.1 row of table 4).
4. **Permutation calibration fixes both problems.** False alarms stay around 3% at every sample size. Weak collusion becomes detectable with enough data (CP = 0.1: 0.17 at 1,000 games, up from 0.04). The cost is lower power at very small n.

## 7. Implications for the market experiment
- A double-auction run has about 30 rounds, which is below even the n = 100 regime here. Thresholds **must** be calibrated, by permutation or on honest runs. A fixed α will not work.
- The state space will be larger than 27 states, so the bias will be larger. This calls for coarse state discretization, pooling across runs, and a sensitivity analysis over the state abstraction.
- Comparing against simple correlation "at the same false-alarm rate" is only fair if both detectors use the same calibration procedure.

## Limitations
- Only one kind of collusion (assistant collusion), 3 players, and scripted (non-LLM) agents.
- Small numbers of seeds (30–50): each cell has a standard error of about ±0.07–0.09.
- The permutation rule tests 3 pairs at α = 0.05 without multiple-testing correction.
- These numbers have not yet been compared with the curves in Bonjour et al.
