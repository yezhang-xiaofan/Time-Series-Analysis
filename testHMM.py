import numpy as np
from sklearn import hmm
from sklearn.hmm import MultinomialHMM

'''
startprob = np.array([0.6, 0.3, 0.1])
transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
covars = np.tile(np.identity(2), (3, 1, 1))
model = hmm.GaussianHMM(3, "full", startprob, transmat)
model.means_ = means
model.covars_ = covars
X, Z = model.sample(100)
model2 = hmm.GaussianHMM(3, "full")
model2.fit([X]) 
#GaussianHMM(algorithm='viterbi',...
Z2 = model.predict(X)
'''

A = np.array([[0.9, 0.1], [0.3, 0.7]])
efair = np.array([1.0 / 6] * 6)
print efair
#[0.16666666666666666, 0.16666666666666666, ... 0.16666666666666666]
eloaded = np.array([3.0 / 13, 3.0 / 13, 2.0 / 13, 2.0 / 13, 2.0 / 13, 1.0 / 13])
B = np.array([efair, eloaded])
pi = np.array([0.5] * 2)

m = hmm.MultinomialHMM(n_components=2, startprob=pi, transmat=A)
m.emissionprob_ = B
my_seq = np.array([0] * 20 + [5] * 10 + [0] * 40)
my_seq = my_seq.reshape(len(my_seq),1)

print my_seq
print m.predict(my_seq)