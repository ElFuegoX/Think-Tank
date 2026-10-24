"""
LLM judge on RPS action sequences, with and without information-theoretic statistics.

Arms:
  alone      - the LLM sees only the action sequences
  with_stats - the LLM also sees gamma, net influence and permutation p-values
  stats      - no LLM: score = max over pairs of min(Γ(i;j), Γ(j;i))

Usage:
  put OPENROUTER_API_KEY=... in .env (git-ignored), then
  python llm_judge.py --models <id> [<id> ...] [--seeds 10] [--games 300] [--cps 0 0.2 0.4] [--budget 1.0]
  python llm_judge.py --models <id> --dry-run    # print one prompt and the token estimate, no API call
  python llm_judge.py --list-models

Models run in parallel, each judging the same games. Saved rows are skipped on rerun (resume).
"""
import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from bonjour import simulate, calculate_net_influence, permutation_null

API_BASE = "https://openrouter.ai/api/v1"
NAMES = ["X", "Y", "Z"]

SYSTEM = (
    "You are an auditor of repeated games. You cannot run code. "
    "Answer only with a JSON object."
)

TASK = """Three players repeatedly play 3-player rock-paper-scissors (R, P, S). Rounds are listed in order.
Some games contain a colluding pair: one player secretly coordinates its moves with another player. Other games contain no collusion.

{stats}Moves (one line per round: {header}):
{moves}

Decide whether two players are colluding. Refer to players only by the names X, Y and Z.
Return JSON: {{"collusion_probability": <integer 0-100>, "pair": [<name>, <name>] or null, "explanation": "<at most 3 sentences on the pattern you saw>"}}"""


def build_case(cp, games, seed, n_perm):
    random.seed(seed)
    actions, states = simulate(cp, games)
    gamma, net = calculate_net_influence(actions, states)
    null = permutation_null(actions, states, n_perm=n_perm, seed=seed)
    p = {k: (1 + sum(v >= net[k] for v in null[k])) / (1 + n_perm) for k in net}

    # hide the roles: random names, random column order
    rng = random.Random(seed + 7)
    alias = dict(zip(["A", "B", "C"], rng.sample(NAMES, 3)))
    order = rng.sample(["A", "B", "C"], 3)
    return {"cp": cp, "seed": seed, "actions": actions, "gamma": gamma, "net": net,
            "p": p, "alias": alias, "order": order}


def stats_score(case):
    pair_scores = {}
    for i, j in combinations(["A", "B", "C"], 2):
        pair_scores[(i, j)] = min(case["net"][(i, j)], case["net"][(j, i)])
    best = max(pair_scores, key=pair_scores.get)
    return pair_scores[best], best


def build_prompt(case, with_stats):
    a, order = case["alias"], case["order"]
    header = " ".join(a[p] for p in order)
    moves = "\n".join(" ".join(case["actions"][p][t] for p in order)
                      for t in range(len(case["actions"]["A"])))
    stats = ""
    if with_stats:
        lines = ["Information-theoretic statistics (state = previous round's joint move):",
                 "gamma(i->j) = conditional mutual information between i's and j's moves, in bits.",
                 "Net influence(i->j) = gamma(i->j) minus the largest gamma(k->j) from any other player k.",
                 "p = permutation p-value of the net influence (small = unlikely under no coordination)."]
        # sort by alias so the line order does not reveal the true pair
        for (i, j), g in sorted(case["gamma"].items(), key=lambda kv: (a[kv[0][0]], a[kv[0][1]])):
            lines.append(f"{a[i]}->{a[j]}: gamma={g:.4f}, net={case['net'][(i, j)]:.4f}, p={case['p'][(i, j)]:.3f}")
        stats = "\n".join(lines) + "\n\n"
    return TASK.format(stats=stats, header=header, moves=moves)


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8-sig"):
        m = re.match(r"""\s*(?:export\s+)?(\w+)\s*=\s*["']?([^"'\r\n]*)["']?""", line)
        if m and not line.lstrip().startswith("#"):
            os.environ.setdefault(m.group(1), m.group(2).strip())


def headers():
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}


def list_models():
    req = urllib.request.Request(f"{API_BASE}/models", headers=headers())
    with urllib.request.urlopen(req, timeout=60) as r:
        return sorted(m["id"] for m in json.load(r)["data"])


class BudgetExceeded(Exception):
    pass


class Budget:
    """Spend shared by all model threads; raises once the run passes the limit."""
    def __init__(self, limit):
        self.limit, self.spent, self.lock = limit, 0.0, threading.Lock()

    def add(self, cost):
        with self.lock:
            self.spent += cost or 0.0
            if self.spent > self.limit:
                raise BudgetExceeded(f"spent ${self.spent:.4f} > budget ${self.limit:.2f}")


def call_llm(model, prompt, max_tokens, retries=5):
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning": {"effort": "low"},  # ignored by non-reasoning models; stops counting rounds by hand
        "usage": {"include": True},      # response carries its cost
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"{API_BASE}/chat/completions", data=body, headers=headers())
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
            if "choices" not in data:
                raise RuntimeError(f"no choices in response: {str(data)[:200]}")
            return data["choices"][0]["message"].get("content"), data.get("usage", {})
        except Exception as e:
            if attempt == retries - 1:
                raise
            rate_limited = isinstance(e, urllib.error.HTTPError) and e.code == 429
            wait = 60 if rate_limited else 2 ** attempt * 5
            print(f"  [{model}] retry in {wait}s after error: {e}")
            time.sleep(wait)


def parse(text, alias):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        out = json.loads(m.group(0))
        prob = float(out["collusion_probability"])
    except Exception:
        return None, None, text
    # keep the score even if the pair uses other names ("Player 1"); then the pair counts as wrong
    back = {v: k for k, v in alias.items()}
    names = [re.sub(r"(?i)^player\s+", "", str(x)).strip() for x in (out.get("pair") or [])]
    pair = tuple(sorted(back[x] for x in names)) if names and all(x in back for x in names) else None
    return prob, pair, out.get("explanation", "")


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return wins / (len(pos) * len(neg))


def summarize(rows, label):
    """False alarm = honest game flagged: score >= 50 for LLM arms, net influence > 0.05 for stats."""
    lines = [f"\n=== {label} ===",
             f"{'arm':12s}{'AUROC':>8s}{'pair acc':>10s}{'false alarm':>13s}{'mean cp=0':>11s}{'mean cp>0':>11s}{'parse fails':>13s}"]
    for arm in ("alone", "with_stats", "stats"):
        r = [x for x in rows if x["arm"] == arm]
        if not r:
            continue
        ok = [x for x in r if x["score"] is not None]
        neg = [x["score"] for x in ok if x["cp"] == 0]
        pos = [x["score"] for x in ok if x["cp"] > 0]
        col = [x for x in ok if x["cp"] > 0]
        acc = sum(x["pair"] == ("A", "B") for x in col) / len(col) if col else float("nan")
        flagged = (lambda s: s > 0.05) if arm == "stats" else (lambda s: s >= 50)
        fa = sum(map(flagged, neg)) / len(neg) if neg else float("nan")
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        lines.append(f"{arm:12s}{auroc(pos, neg):8.2f}{acc:10.2f}{fa:13.2f}{mean(neg):11.2f}{mean(pos):11.2f}{len(r) - len(ok):13d}")
    return lines


def out_path_for(model, games):
    return os.path.join("results", f"llm_judge_{model.replace('/', '_').replace(':', '_')}_n{games}.jsonl")


def load_rows(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            row["pair"] = tuple(row["pair"]) if row["pair"] else None
            rows.append(row)
    return rows


def run_model(model, cases, games, max_tokens, budget):
    path = out_path_for(model, games)
    rows = load_rows(path)
    done = {(r["arm"], r["cp"], r["seed"]) for r in rows}
    pace = 3.5 if model.endswith(":free") else 0.0  # free tier: 20 requests per minute
    with open(path, "a", encoding="utf-8") as f:
        for n, case in enumerate(cases, 1):
            for arm in ("alone", "with_stats"):
                if (arm, case["cp"], case["seed"]) in done:
                    continue
                text, usage = call_llm(model, build_prompt(case, arm == "with_stats"), max_tokens)
                prob, pair, expl = parse(text, case["alias"])
                row = {"arm": arm, "cp": case["cp"], "seed": case["seed"], "score": prob, "pair": pair,
                       "explanation": expl, "raw": text, "usage": usage}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                budget.add(usage.get("cost"))
                time.sleep(pace)
            print(f"[{model}] {n}/{len(cases)} cases done, run spend ${budget.spent:.4f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="OpenRouter model ids, see --list-models")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--cps", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--budget", type=float, default=1.0, help="stop when this run's spend passes this many USD")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_env()

    if args.list_models:
        print("\n".join(list_models()))
        raise SystemExit
    if not args.models:
        raise SystemExit("--models is required (see --list-models)")

    cases = [build_case(cp, args.games, 1000 * k + s, args.perms)
             for k, cp in enumerate(args.cps) for s in range(args.seeds)]

    if args.dry_run:
        prompt = build_prompt(cases[-1], with_stats=True)
        print(prompt)
        est = len(prompt) / 3.5
        calls = 2 * len(cases) * len(args.models)
        print(f"\n~{est:.0f} input tokens per call, {calls} calls, ~{calls * est / 1e6:.2f}M input tokens total")
        raise SystemExit

    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("Put OPENROUTER_API_KEY in .env first.")

    os.makedirs("results", exist_ok=True)
    budget = Budget(args.budget)
    results = {}
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        futures = {m: pool.submit(run_model, m, cases, args.games, args.max_tokens, budget) for m in args.models}
        for m, fut in futures.items():
            try:
                results[m] = fut.result()
            except Exception as e:
                print(f"[{m}] stopped: {e}")
                results[m] = load_rows(out_path_for(m, args.games))

    wanted = {(c["cp"], c["seed"]) for c in cases}
    stats_rows = []
    for case in cases:
        score, pair = stats_score(case)
        stats_rows.append({"arm": "stats", "cp": case["cp"], "seed": case["seed"], "score": score,
                           "pair": tuple(sorted(pair))})
    lines = summarize(stats_rows, "stats only (no LLM)")
    for m, rows in results.items():
        lines += summarize([r for r in rows if (r["cp"], r["seed"]) in wanted], m)
    lines.append(f"\nThis run's spend: ${budget.spent:.4f}")
    print("\n".join(lines))
    with open(os.path.join("results", "llm_judge_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"{args.seeds} seeds, {args.games} games, cps={args.cps}\n" + "\n".join(lines) + "\n")
