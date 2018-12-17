import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def vis_dist_data():


    episodes_per_eucl_dist = np.array([26, 26, 7, 14, 2, 2, 8, 3, 1, 1])
    n_episodes = np.sum(episodes_per_eucl_dist)
    data_hist = np.zeros(n_episodes)
    for i, freq in enumerate(episodes_per_eucl_dist):
        for f in range(freq):
            data_hist[np.sum(episodes_per_eucl_dist[:i])+f] = i

    episodes_per_eucl_dist_posterior = np.array([1, 1, 7, 13, 16, 9, 8, 5, 3, 26])
    data_hist_post = np.zeros(np.sum(episodes_per_eucl_dist_posterior))
    for i, freq in enumerate(episodes_per_eucl_dist_posterior):
        for f in range(freq):
            data_hist_post[np.sum(episodes_per_eucl_dist_posterior[:i])+f] = i

    data_hist = data_hist_post
    
    sns.set_style("darkgrid")
    sns.set_style("whitegrid")
    plt.hist(data_hist, alpha=0.7, rwidth=0.85, bins=10)
    sns.set_style("ticks")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlim([0, 9])
    plt.xlabel(r"\textbf{Rating ($0$: No collision, $9$: Collision)}", size=14)
    plt.ylabel(r"$\textbf{Frequency}$", size=14)
    plt.title(r"\textbf{Histogram of HMM prediction (Total $89$ Samples)}$", size=15)
    plt.show()

vis_dist_data()