
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# load and preprocess the data

spambase_path = "../data/spambase/spambase.data"
X = np.genfromtxt(spambase_path, delimiter=',')

Y = X[:, -1]
## modify 10% data labels
tochange = np.random.choice(Y.shape[0], int(0.1*Y.shape[0]))
Y[tochange] = 1 - Y[tochange]

X = X[:, :-1]
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = np.hstack((X, np.ones((X.shape[0],1))))

Y[Y < 0.01] = -1


# In[3]:


def perceptron(X, Y, T, eta=1.0):
    correct = np.zeros(T,)

    N, d = X.shape
    W = np.zeros(d,)

    for t in range(T):
        it = np.random.randint(N)
        x = X[it]

        y_bar = -1 if W.dot(x) < 0 else 1
        y = Y[it]

        if y*y_bar < 0:
            W = W + eta * y * x
#             W = W + eta * np.sqrt(1.0/(t+1)) * y * x
            correct[t] = 0
        else:
            correct[t] = 1

    return correct.cumsum() / np.arange(1, T+1)


# In[4]:


def winnow(X, Y, T, eta=1.0):
    correct = np.zeros(T,)
    X = np.hstack((X, -X))

    N, d = X.shape
    W = np.ones(d,) / d

    for t in range(T):
        it = np.random.randint(N)
        x = X[it]

        y_bar = -1 if W.dot(x) < 0 else 1
        y = Y[it]

        if y*y_bar < 0:
#             W = W * np.exp(eta * y * x)
            W = W * np.exp(eta*np.sqrt(1.0/(t+1)) * y * x)
            W = W / W.sum()
            correct[t] = 0
        else:
            correct[t] = 1

    return correct.cumsum() / np.arange(1, T+1)



# In[5]:


T = 10000
runs = 100
correct_perc1 = np.zeros((runs, T))

for i in range(runs):
    np.random.seed(i*57)
    cor = perceptron(X, Y, T, eta=1.0)
    correct_perc1[i, :] = 1-cor


# In[6]:


T = 10000
runs = 100
correct_winn1 = np.zeros((runs, T))

for i in range(runs):
    np.random.seed(i*57)
    cor = winnow(X, Y, T, eta=0.01)
    correct_winn1[i, :] = 1-cor


# In[7]:


# get_ipython().magic(u'matplotlib notebook')

xlab = np.arange(T)
g = 100


mu = correct_perc1.mean(axis=0)
sig = correct_perc1.std(axis=0) * 3
plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='o', label="perceptron")
# plt.title("Perceptron on Spambase database")

mu = correct_winn1.mean(axis=0)
sig = correct_winn1.std(axis=0) * 3
plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='o', label="winnow")
# plt.title("Winnow on Spambase database")


plt.legend(loc=0)
plt.ylim(0, 0.5)
plt.ylabel("wrong predictions")
plt.xlabel("time steps")
plt.show()
