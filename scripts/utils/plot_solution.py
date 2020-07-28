import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def plot_solution(sol, n_rod, l, rot=None):
    '''
    plot chain links in 2D
    '''
    colors = sns.hls_palette(n_rod, l=.3, s=.8) 
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for i in range(n_rod):
        xc, yc, theta = sol[i*3:i*3+3]
        rot_mtx = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])
        bar = np.dot(rot_mtx, [[-0.5*l[i], 0.5*l[i]], [0, 0]]) + np.array([[xc], [yc]])
        ax.plot(bar[0], bar[1], color=colors[i], lw=2.)
    ax.axis('equal')
        
    plt.show()