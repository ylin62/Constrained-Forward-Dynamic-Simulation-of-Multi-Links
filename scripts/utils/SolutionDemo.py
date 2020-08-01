import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib
import seaborn as sns
from IPython.display import display, Markdown, HTML
sns.set_style("whitegrid")

LW=2
FONTSIZE=18
FIGSIZE=(12, 8)
MARKER = 10
ARROWLENGTH = 2.5
HW = 0.8

def print_constrains(Demo):
    display(Markdown("### System constrains:"))
    constrains = Demo.constrains()
    for item in constrains:
        display(sp.Eq(item, 0))
        
def print_govs(Model, f):
    '''
    print governing equations without damping for now
    '''
    display(Markdown("### System governing equations"))
    L = Model.lagrangian()
    left = (L.jacobian(Model.q_dot)).diff(Model.t)
    right = L.jacobian(Model.q)
    
    for i, item in enumerate(left):
        display(sp.Eq(item, right[i] + f[i]))
        
def get_multipliers(model, f, g, sol, show=False):
    
    multips = []
    for y in sol.y.T:
        input_f = np.append(y, [f, model.M.diagonal() * g])
        a = model.a(*y[:3*model.n_rod])
        b = model.b(*input_f)
        multips += [-np.linalg.solve(a, b)[3*model.n_rod:]]
    
    multips = np.concatenate(multips, axis=1)
    
    if show:
        colors = sns.hls_palette(len(multips), l=.3, s=.8)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.xaxis.offsetText.set_fontsize(FONTSIZE)
        ax.yaxis.offsetText.set_fontsize(FONTSIZE)
        
        for i, item in enumerate(multips):
            ax.plot(sol.t, item, lw=LW, color=colors[i], label='$\\lambda_{'+str(i+1)+'}$')
        ax.set_xlabel('T [s]', fontsize=FONTSIZE)
        ax.set_ylabel('$\\lambda$', fontsize=FONTSIZE)
        fig.legend(fontsize=FONTSIZE)
        fig.suptitle('$Lagrangian$ multipliers', fontsize=FONTSIZE)
        
        plt.show()
        
    return multips

class SolutionDemo(object):
    """docstring for ClassName"""
    def __init__(self, sol, m, l, rot=None):
        super(SolutionDemo, self).__init__()
        self.t = sol.t
        self.y = sol.y
        self.m = m
        self.n_rod = len(m)
        self.l = l[:len(m)]
        self.rot = rot
        
    def plot_solution(self, idx):
        '''
        plot chain links in 2D
        '''
        colors = sns.hls_palette(self.n_rod, l=.3, s=.8) 
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        for i in range(self.n_rod):
            xc, yc, theta = self.y[i*3:i*3+3, idx]
            rot_mtx = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
            bar = np.dot(rot_mtx, [[-0.5*self.l[i], 0.5*self.l[i]], [0, 0]]) + np.array([[xc], [yc]])
            
            if self.rot is not None:
                rot_plot = np.array([[np.cos(self.rot), -np.sin(self.rot)], 
                                     [np.sin(self.rot), np.cos(self.rot)]])
                bar = np.dot(rot_plot, bar)
            
            ax.plot(bar[0], bar[1], 'o-', color=colors[i], lw=2., label='$link_{'+str(i)+'}$')
            
        ax.axis('equal')
        ax.set_xlabel('X [m]', fontsize=FONTSIZE)
        ax.set_ylabel('Y [m]', fontsize=FONTSIZE)
        fig.legend(fontsize=FONTSIZE)
        fig.suptitle('links at time {:.2f}s'.format(self.t[idx]), fontsize=FONTSIZE)        
        for axe in [ax]:
            axe.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            axe.xaxis.offsetText.set_fontsize(FONTSIZE)
            axe.yaxis.offsetText.set_fontsize(FONTSIZE)
            
        plt.show()
    
    @property
    def links(self):
        
        if self.rot is not None:
            rot_plot = np.array([[np.cos(self.rot), -np.sin(self.rot)], 
                                 [np.sin(self.rot), np.cos(self.rot)]])
        
        links = []
        for i in range(self.n_rod):
            xc, yc, theta = self.y[i*3:i*3+3]
            bar = []
            for x, y, h in zip(xc, yc, theta):
                rot_mtx = np.array([[np.cos(h), -np.sin(h)], 
                                    [np.sin(h), np.cos(h)]])
                if self.rot is None:
                    bar += [np.dot(rot_mtx, [[-0.5*self.l[i], 0.5*self.l[i]], [0, 0]]) + np.array([[x], [y]])]
                else:
                    temp = np.dot(rot_mtx, [[-0.5*self.l[i], 0.5*self.l[i]], [0, 0]]) + np.array([[x], [y]])
                    bar += [np.dot(rot_plot, temp)]
            links += [np.array(bar)]
            
        return np.array(links)

    def play_IPython(self, interval=50, title=None, save_as=None):
        
        lim = sum(self.l)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.xaxis.offsetText.set_fontsize(FONTSIZE)
        ax.yaxis.offsetText.set_fontsize(FONTSIZE)
        ax.set_xlim(( -lim, lim))
        ax.set_ylim(( -lim, lim))
        if title is not None:
            fig.suptitle(title, fontsize=FONTSIZE)
        line, = ax.plot([], [], 'o-', lw=LW)
        time_template = 'time = {:.2f}s'
        time_text = ax.text(0.05, 0.9, '', fontsize=FONTSIZE, transform=ax.transAxes)

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            x = self.links[:, i, 0]
            y = self.links[:, i, 1]
            line.set_data(x, y)
            time_text.set_text(time_template.format(self.t[i]))
            return line, time_text
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                       frames=self.links.shape[1], 
                                       interval=interval,
                                       blit=True)
        plt.close(anim._fig)
        
        if save_as:
            folder = os.path.split(save_as)[0]
            if len(folder) > 0:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            anim.save(save_as, writer='ffmpeg')
        
        return HTML(anim.to_jshtml())
    
    def plot_angles(self):
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.xaxis.offsetText.set_fontsize(FONTSIZE)
        ax.yaxis.offsetText.set_fontsize(FONTSIZE)
        
        for i in range(self.n_rod):
            ax.plot(self.t, self.y[i*3+2], lw=LW, label='$\\theta_{'+str(i+1)+'}$')
        ax.set_xlabel('T [s]', fontsize=FONTSIZE)
        ax.set_ylabel('Angle [rads]', fontsize=FONTSIZE)
        fig.legend(fontsize=FONTSIZE)
        
        plt.show()
        