import random
import math
import numpy as np
import matplotlib.pyplot as plt

def bernoulli(p):
    return float(random.random() < p)

alpha = 4.0

def ucb(T, P, rewards=None):
    N = len(P)
    if rewards is None:
        rewards = np.array([[bernoulli(p) for i in range(T)] for p in P]).astype(np.float)

    history = np.zeros(T,).astype(np.int)

    running_sum = np.zeros(N,).astype(np.float)
    N_i = np.zeros(N,).astype(np.float)

    for i in range(N):
        history[i] = i
        running_sum[i] = rewards[i, i]
        N_i[i] = 1

    for t in range(N, T):
        upper = running_sum / N_i + np.sqrt(1.0 * alpha * np.log(t+1) / (2 * N_i))
        it = np.argmax(upper)

        running_sum[it] += rewards[it, t]
        N_i[it] += 1
        history[t] = it

    # mu for bernoulli is p
    i_star = np.argmax(P)
    regret = np.array([rewards[i_star, idx]-rewards[it, idx] for (idx,it) in enumerate(history)])
    regret = regret.cumsum()

    is_optimal = np.array([float(it == i_star) for it in history])
    is_optimal = is_optimal.cumsum() / np.arange(1, T+1)

    return {
        'history' : history,
        'regret' : regret,
        'is_optimal' : is_optimal
        }

def main():
    num_runs = 100
    T = 10000
    P = (0.55, 0.45)
    is_optimal = np.zeros((num_runs, T))
    regret = np.zeros((num_runs, T))
    random.seed(37)

    for i in range(num_runs):
        res = ucb(T=T, P=P)
        is_optimal[i, :] = res['is_optimal']
        regret[i, :] = res['regret']

    xlab = np.arange(T)
    g = 100

    mu = regret.mean(axis=0)
    sig = regret.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="UCB {}".format(str(P)))
    plt.title("Regret")

    plt.legend(loc=0)
    # plt.ylim(0, 1)
    plt.ylabel("regret")
    plt.xlabel("time steps")
    plt.show()



    mu = is_optimal.mean(axis=0)
    sig = is_optimal.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="UCB {}".format(str(P)))
    plt.title("Fraction of times optimal arm played")

    plt.legend(loc=0)
    # plt.ylim(0, 1)
    plt.ylabel("fraction of times optimal arm played")
    plt.xlabel("time steps")
    plt.show()



if __name__ == "__main__":
    main()
