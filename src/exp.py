
# import sys
# print (sys.version)
# print (‘hello world’)
# Hidden parameter: euclidean distance
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt


# example of input 
a1 = np.array([[0,0], [0,1], [0,2], [0,3], [0,4]])
a2 = np.array([[4, 0],  [3, 0], [2, 0], [1, 0], [0, 0]])


## Experiments

# positions
a1 = np.array([[0,0], [0,1], [0,2], [0,3], [0,4]])
a2 = np.array([[4, 0], [3, 0], [2, 0], [1, 0], [0, 0]])

dir = np.array([[0,1],[1,0],[-1,0],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]])

# directions
# for i in range(1, len(a1)):
# 	delta = (a1[i]-a1[i-1]).tolist()
# 	dir1.append(delta)
# 	print(delta.tolist(), type(delta))

# res = create_multisequence_model(dir,dir)

# print(res)
# print(sample_model(res, 1))

dir_index = 3
# this is supposed to be a probability disitribution over components
cur_dir = np.zeros(8)
cur_dir[dir_index] = 1


model_exp = hmm.GaussianHMM(n_components=8, covariance_type="full")

# Build an HMM instance and set parameters
# 8 signifies the 8 diff options on directions
model = hmm.GaussianHMM(n_components=4, covariance_type="full")



startprob = np.array([0.6, 0.3, 0.1, 0.0])
# The transition matrix, note that there are no transitions possible
# between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])
# # The means of each component
means = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])
# The covariance of each component
covars = .5 * np.tile(np.identity(2), (4, 1, 1))


# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars
# Generate samples
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