
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from scipy import sparse

UNK = "<unk>"
smspath = "../data/smsspamcollection/SMSSpamCollection"

class Vocabulary(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
        self.cnt = 0
    def add(self, w):
        self.word2idx[w] = self.cnt
        self.idx2word[self.cnt] = w
        self.cnt += 1

    def __call__(self, w):
        if w in self.word2idx:
            return self.word2idx[w]
        return self.word2idx[UNK]

    def __len__(self):
        return self.cnt

with open(smspath, "r") as f:
    lines = f.readlines()

Y = []
X_raw = []
count = defaultdict(int)

for idx, sms in enumerate(lines):
    sms = re.sub('[^A-Za-z]', ' ', sms).lower().split()
    y = -1 if sms[0] == "spam" else 1

    Y.append(y)
    X_raw.append(sms[1:])

    for w in sms[1:]:
        count[w] += 1

Y = np.array(Y)


# In[2]:


word_freq = count.items()
total_words = sum([c for w, c in word_freq])

thresh = total_words*0.01
vocab_words = [w for w, c in word_freq if c < thresh]

vocab = Vocabulary()
vocab.add(UNK)
for w in vocab_words:
    vocab.add(w)


# In[3]:


X = np.zeros((len(X_raw), len(vocab)))

for idx, x in enumerate(X_raw):
    freq = defaultdict(int)
    for w in x:
        freq[w] += 1

    for w, c in freq.items():
        X[idx, vocab(w)] = c


# In[4]:


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


# In[5]:


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


# In[ ]:


T = 10000
runs = 100
correct_perc1 = np.zeros((runs, T))

for i in range(runs):
    np.random.seed(i*57)
    cor = perceptron(X, Y, T, eta=1.0)
    correct_perc1[i, :] = 1-cor


# In[ ]:


T = 10000
runs = 100
correct_winn1 = np.zeros((runs, T))

for i in range(runs):
    np.random.seed(i*57)
    cor = winnow(X, Y, T, eta=0.01)
    correct_winn1[i, :] = 1-cor


# In[ ]:


# get_ipython().magic(u'matplotlib notebook')

xlab = np.arange(T)
g = 100

mu = correct_perc1.mean(axis=0)
sig = correct_perc1.std(axis=0) * 5
plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='o', label="perceptron")
# plt.title("Perceptron on Spambase database")

mu = correct_winn1.mean(axis=0)
sig = correct_winn1.std(axis=0) * 3
plt.errorbar(xlab[::g], mu[::g], yerr=sig[::g], fmt='o', label="winnow")
# plt.title("Winnow on Spambase database")


plt.legend(loc=0)
plt.ylim(0, 0.3)
plt.ylabel("correct predictions")
plt.xlabel("time steps")
plt.show()
