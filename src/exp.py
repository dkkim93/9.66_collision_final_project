
# import sys
# print (sys.version)
# print (‘hello world’)
# Hidden parameter: euclidean distance
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt


## Experiments

# get directions by time steps
def get_direction(pos):
	"""
	pos is a n x 2 vector
	"""
	dirs = []

	for i in range(1, len(pos)):
		delta1 = (pos[i]-pos[i-1])
		dirs.append(delta1.tolist())

	return dirs


# get direction vector
def get_direction_vector(directions):
	"""
	dir = [1,0]
	"""
	to_be_returned = []
	for i in range(len(directions)):
		res = np.zeros(8)
		order = np.array([[0,1],[1,1],[1,0], [1,-1], [0,-1], [-1,-1],[-1,0], [-1,1]])
		ind = np.where((order==tuple(directions[i])).all(axis=1))
		res[ind] = 1
		to_be_returned.append(res)
	return to_be_returned


# positions
a1 = np.array([[0,0], [0,1], [0,2], [0,3], [0,4]])
a2 = np.array([[4, 0], [3, 0], [2, 0], [1, 0], [0, 0]])

# dir1 = get_direction_vector(get_direction(a1))
# dir2 = get_direction_vector(get_direction(a2))


dir1 = get_direction(a1)
dir2 = get_direction(a2)

model_exp = hmm.GaussianHMM(n_components=8, covariance_type="full")
startprob = np.ones(model_exp.n_components) / model_exp.n_components
transmat = np.ones((model_exp.n_components,model_exp.n_components)) / model_exp.n_components
means = np.array([[0,1],[1,1],[1,0], [1,-1], [0,-1], [-1,-1],[-1,0], [-1,1]])
covars = .5 * np.tile(np.identity(2), (8, 1, 1))

model_exp.startprob_ = startprob
model_exp.transmat_ = transmat
model_exp.means_ = means
model_exp.covars_ = covars


# print(dir1)
# model_exp.fit(dir1)

X, Z = model_exp.sample(500)

print(X)

# Plot the sampled data
plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
         mfc="orange", alpha=0.7)

# Indicate the component numbers
for i, m in enumerate(means):
    plt.text(m[0], m[1], 'Component %i' % (i + 1),
             size=17, horizontalalignment='center',
             bbox=dict(alpha=.7, facecolor='w'))
plt.legend(loc='best')
plt.show()


X, Z = model.sample(500)

# Plot the sampled data
plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
         mfc="orange", alpha=0.7)

# Indicate the component numbers
for i, m in enumerate(means):
    plt.text(m[0], m[1], 'Component %i' % (i + 1),
             size=17, horizontalalignment='center',
             bbox=dict(alpha=.7, facecolor='w'))
plt.legend(loc='best')
plt.show()
