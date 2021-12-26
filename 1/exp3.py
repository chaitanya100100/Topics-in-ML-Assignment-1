import random
import math
import numpy as np
import matplotlib.pyplot as plt

def bernoulli(p):
    return float(random.random() < p)

def exp3(T, P, rewards=None):
    N = len(P)
    if rewards is None:
        rewards = np.array([[bernoulli(p) for i in range(T)] for p in P]).astype(np.float)
    losses = -rewards # we need notion of loss instead of rewards

    # eta = 0.0083
    eta = np.sqrt(2.0 * np.log(N) / (T * N))

    N_i = np.array([0 for i in range(N)])
    history = np.zeros(T,).astype(np.int)

    weights = np.ones(N,).astype(np.float) / N
    L = np.zeros(N,).astype(np.float)

    for t in range(T):
        it = np.random.choice(np.arange(N), p=weights)
        N_i[it] += 1
        history[t] = it
        L[it] = L[it] + losses[it, t] / weights[it]

        weights = np.exp(-eta*L)
        weights = weights / np.sum(weights)

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
        res = exp3(T=T, P=P)
        is_optimal[i, :] = res['is_optimal']
        regret[i, :] = res['regret']

    xlab = np.arange(T)
    g = 100

    mu = regret.mean(axis=0)
    sig = regret.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="EXP3 {}".format(str(P)))
    plt.title("Regret")

    plt.legend(loc=0)
    # plt.ylim(0, 1)
    plt.ylabel("regret")
    plt.xlabel("time steps")
    plt.show()



    mu = is_optimal.mean(axis=0)
    sig = is_optimal.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="EXP3 {}".format(str(P)))
    plt.title("Fraction of times optimal arm played")

    plt.legend(loc=0)
    # plt.ylim(0, 1)
    plt.ylabel("fraction of times optimal arm played")
    plt.xlabel("time steps")
    plt.show()



if __name__ == "__main__":
    main()
