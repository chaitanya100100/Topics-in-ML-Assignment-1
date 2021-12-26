from eps_greedy import eps_greedy
from ucb import ucb
from exp3 import exp3
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def bernoulli(p):
    return float(np.random.rand() < p)


def main():
    num_runs = 10
    T = 10000
    P=(0.9, 0.8)

    io_ucb = np.zeros((num_runs, T))
    r_ucb = np.zeros((num_runs, T))

    io_exp3 = np.zeros((num_runs, T))
    r_exp3 = np.zeros((num_runs, T))

    io_eg1 = np.zeros((num_runs, T))
    r_eg1 = np.zeros((num_runs, T))

    io_eg2 = np.zeros((num_runs, T))
    r_eg2 = np.zeros((num_runs, T))

    for i in range(num_runs):
        random.seed(37*i*i+31*i+29)
        rewards = np.array([[bernoulli(p) for i in range(T)] for p in P]).astype(np.float)

        res = ucb(T=T, P=P, rewards=rewards)
        io_ucb[i, :] = res['is_optimal']
        r_ucb[i, :] = res['regret']

        res = exp3(T=T, P=P, rewards=rewards)
        io_exp3[i, :] = res['is_optimal']
        r_exp3[i, :] = res['regret']

        res = eps_greedy(T=T, P=P, eps=0.01, rewards=rewards)
        io_eg1[i, :] = res['is_optimal']
        r_eg1[i, :] = res['regret']

        res = eps_greedy(T=T, P=P, eps=0.1, rewards=rewards)
        io_eg2[i, :] = res['is_optimal']
        r_eg2[i, :] = res['regret']


    xlab = np.arange(T)
    g = 100

    mu = r_ucb.mean(axis=0)
    sig = r_ucb.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="UCB")

    mu = r_exp3.mean(axis=0)
    sig = r_exp3.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="EXP3")

    mu = r_eg1.mean(axis=0)
    sig = r_eg1.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="epsGreedy (eps=0.01)")

    mu = r_eg2.mean(axis=0)
    sig = r_eg2.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="epsGreedy (eps=0.1)")

    plt.legend(loc=0)
    plt.title("Regret")
    plt.ylabel("regret")
    plt.xlabel("time steps")
    plt.show()

    mu = io_ucb.mean(axis=0)
    sig = io_ucb.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="UCB")

    mu = io_exp3.mean(axis=0)
    sig = io_exp3.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="EXP3")

    mu = io_eg1.mean(axis=0)
    sig = io_eg1.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="epsGreedy (eps=0.01)")

    mu = io_eg2.mean(axis=0)
    sig = io_eg2.std(axis=0)
    plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='-o', label="epsGreedy (eps=0.1)")


    plt.legend(loc=0)
    plt.title("Fraction of times optimal arm played")
    plt.ylabel("fraction of times optimal arm played")
    plt.xlabel("time steps")
    plt.show()



if __name__ == "__main__":
    main()
