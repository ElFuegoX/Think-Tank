"""
Pairwise collusion detectors shared by four_agent.py, market.py and double_auction.py.

Every pair detector takes {player: action sequence} (+ optional states) and returns {(i, j): score}
with i < j, higher = more suspicious. Run score = max over pairs; thresholds are calibrated on
honest runs (see calibrate / evaluate_detectors).

  net_influence   Bonjour et al. net influence, min over the two directions          (tool: baseline method)
  cramers_v       ordinary action association (Cramér's V), no state conditioning    (baseline)
  transfer_entropy max over directions of I(a_i[t-1]; a_j[t] | a_j[t-1])            (tool 2)
  compression_mi  C(x) + C(y) - C(x,y) with lzma, a state-free MI estimate           (tool 6)
  synergy         I(T; X, Y) - I(T; X) - I(T; Y) for triples (interaction info)     (tool 3)
  sprt_alarm_time first round at which a running log-likelihood ratio crosses a bound (tool 5)
"""
import lzma
import math
from collections import Counter
from itertools import combinations

import numpy as np

from bonjour import calculate_individual_influence, calculate_net_influence


def pairs_of(players):
    return list(combinations(sorted(players), 2))


def entropy(seq):
    n = len(seq)
    return -sum(c / n * math.log2(c / n) for c in Counter(seq).values())


def mutual_information(x, y):
    return entropy(x) + entropy(y) - entropy(list(zip(x, y)))


def net_influence(actions, states=None):
    _, net = calculate_net_influence(actions, states)
    return {(i, j): min(net[(i, j)], net[(j, i)]) for i, j in pairs_of(actions)}


def cramers_v(actions, states=None):
    out = {}
    for i, j in pairs_of(actions):
        x, y = actions[i], actions[j]
        n = len(x)
        rx, ry = sorted(set(x)), sorted(set(y))
        if len(rx) < 2 or len(ry) < 2:
            out[(i, j)] = 0.0
            continue
        joint = Counter(zip(x, y))
        cx, cy = Counter(x), Counter(y)
        chi2 = sum((joint[(a, b)] - cx[a] * cy[b] / n) ** 2 / (cx[a] * cy[b] / n) for a in rx for b in ry)
        out[(i, j)] = math.sqrt(chi2 / (n * (min(len(rx), len(ry)) - 1)))
    return out


def _te(src, dst):
    """I(src[t-1]; dst[t] | dst[t-1]) in bits."""
    past_s, past_d, now_d = src[:-1], dst[:-1], dst[1:]
    return calculate_individual_influence(past_s, now_d, past_d)


def transfer_entropy(actions, states=None):
    return {(i, j): max(_te(actions[i], actions[j]), _te(actions[j], actions[i])) for i, j in pairs_of(actions)}


def _clen(symbols):
    return len(lzma.compress(bytes(symbols), preset=9))


def compression_mi(actions, states=None):
    """Compression estimate of shared information: C(x) + C(y) - C(xy), xy = per-round joint symbol."""
    codes = {}
    enc = {p: [codes.setdefault(a, len(codes)) for a in seq] for p, seq in actions.items()}
    k = max(len(codes), 1)
    out = {}
    for i, j in pairs_of(actions):
        joint = [a * k + b for a, b in zip(enc[i], enc[j])]
        if max(joint) > 255:
            raise ValueError("too many action symbols for byte encoding")
        out[(i, j)] = _clen(enc[i]) + _clen(enc[j]) - _clen(joint)
    return out


def synergy(actions, states=None):
    """For each triple, max over targets T of I(T; X,Y) - I(T;X) - I(T;Y). Returns {(i, j, k): score}."""
    out = {}
    for trio in combinations(sorted(actions), 3):
        best = -1e9
        for t in trio:
            x, y = [p for p in trio if p != t]
            T, X, Y = actions[t], actions[x], actions[y]
            best = max(best, mutual_information(T, list(zip(X, Y))) - mutual_information(T, X) - mutual_information(T, Y))
        out[trio] = best
    return out


def sprt_alarm_time(scores_by_round, threshold):
    """First round index at which the running statistic exceeds threshold, or None."""
    for t, s in enumerate(scores_by_round):
        if s > threshold:
            return t
    return None


DETECTORS = {
    "net_influence": net_influence,
    "cramers_v": cramers_v,
    "transfer_entropy": transfer_entropy,
    "compression_mi": compression_mi,
}


def run_score(pair_scores):
    best = max(pair_scores, key=pair_scores.get)
    return pair_scores[best], best


def calibrate(honest_scores, fpr=0.05):
    return float(np.quantile(honest_scores, 1 - fpr))


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
