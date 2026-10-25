import random
from bonjour import simulate, calculate_net_influence, permutation_null,simulate_one_round

# 置换检验版本的扫描：当 (i, j) 两个方向的 p 值都小于 ALPHA 时判定合谋
ALPHA, N_SEEDS, N_PERM = 0.05, 30, 100
STRIDE = 10

if __name__ == "__main__":
    print(f"permutation rule: pair flagged if p<{ALPHA} in both directions; {N_SEEDS} seeds, {N_PERM} perms")
    print("CP     n=100     n=300     n=1000")
    for cp in (0.1, 0.2, 0.4):
        row = []
        for n in (100, 300, 1000):
            hits = 0
            for seed in range(N_SEEDS):
                random.seed(seed)
                actions={'A':[],'B':[],'C':[]}
                states=[]
                for i in range(n):
                    act,state = simulate_one_round(cp)
                    actions['A'].append(act[0])
                    actions['B'].append(act[1])
                    actions['C'].append(act[2])
                    states.append(state)
                    if (i + 1) % STRIDE == 0 or i == n - 1:
                        _, net = calculate_net_influence(actions, states)
                        null = permutation_null(actions, states, n_perm=N_PERM, seed=seed)
                        p = {k: (1 + sum(v >= net[k] for v in null[k])) / (1 + N_PERM) for k in net}
                        found = [(i, j) for (i, j) in net if i < j and p[(i, j)] < ALPHA and p[(j, i)] < ALPHA]
                        hits += (found == [("A", "B")]) if cp > 0 else (len(found) > 0)
                        if len(found)>0:
                            print(f"cp={str(cp)}, found:"+ str(found[0])+"at round:"+str(i))
                            break
                print("seed:",seed)
                print("n:",n)
                print("one round over")
            row.append(hits / N_SEEDS)
        print(f"{cp:<7.2f}" + "".join(f"{r:<10.2f}" for r in row))
            

                    
                