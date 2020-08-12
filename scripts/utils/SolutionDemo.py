import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import seaborn as sns
from IPython.display import display, Markdown, HTML

LW=2
FONTSIZE=18
FIGSIZE=(10, 6)
MARKER = 10
ARROWLENGTH = 2.5
HW = 0.8

def print_constrains(Demo):
    display(Markdown("### System constrains:"))
    for item in Demo.constrains:
        display(sp.Eq(item, 0))
        
def print_govs(Model, f):
    '''
    print governing equations without damping for now
    '''
    display(Markdown("### System governing equations"))
    L = Model.lagrangian
    left = (L.jacobian(Model.q_dot)).diff(Model.t)
    right = L.jacobian(Model.q)
    
    for i, item in enumerate(left):
        display(sp.Eq(item, right[i] + f[i]))
        
def get_multipliers(model, f, g, sol, show=False):
    
    a, b = model.get_multipliers()
    multips = []
    for y in sol.y.T:
        input_f = np.append(y, [f, g])
        A = a(*y[:3*model.n_rod])
        B = b(*input_f)
        multips += [-np.linalg.solve(A, B)[3*model.n_rod:]]
    multips = np.concatenate(multips, axis=1)
    
    if show:
        sns.set_style("whitegrid")
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
        
    def plot_solution(self, idx):
        '''
        plot chain links in 2D
        '''
        sns.set_style("whitegrid")
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
    
    def plot_angles(self):
        
        sns.set_style("whitegrid")
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
        
    def animate(self, interval=100, title=None, figsize=(6, 6), axis='off', save_as=None):
        
        sns.set_style('white')
        lim = sum(self.l)
        fig, ax = plt.subplots(figsize=figsize)
        if axis == 'off':
            ax.set_axis_off()
            dpi = 100
        else:
            dpi = 150
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax.xaxis.offsetText.set_fontsize(FONTSIZE)
            ax.yaxis.offsetText.set_fontsize(FONTSIZE)
            time_template = 'time = {:.2f}s'
            time_text = ax.text(0.05, 0.9, '', fontsize=FONTSIZE, transform=ax.transAxes)
            if title is not None:
                fig.suptitle(title, fontsize=FONTSIZE)
        ax.set_xlim(( -lim, lim))
        ax.set_ylim(( -lim, lim))
        line, = ax.plot([], [], 'o-', lw=LW, markersize=MARKER)
        ax.set_aspect('equal', adjustable='box')

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            x = self.links[:, i, 0]
            y = self.links[:, i, 1]
            line.set_data(x, y)
            if axis == 'off':
                return line,
            else:
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
            
            if save_as.split('.')[-1] == 'gif':        
                writer = animation.PillowWriter(fps=20)
                anim.save(save_as, writer=writer, dpi=dpi)
            elif save_as.split('.')[-1] == 'mp4':
                anim.save(save_as, writer='ffmpeg', dpi=dpi)
            else:
                print('only support gif or mp4')
        
        return HTML(anim.to_jshtml())
    
    def more_anim(self,  multipliers, interval=100, figsize=(18, 10), title=None, save_as=None):
        
        from matplotlib.gridspec import GridSpec
        
        colors = sns.hls_palette(len(multipliers), l=.3, s=.8)
        sns.set_style('white')
        lim = sum(self.l)
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = fig.add_gridspec(2, 3)
        ax1 = fig.add_subplot(gs[:, :2])
        ax2 = fig.add_subplot(gs[0, -1])
        ax3 = fig.add_subplot(gs[1:, -1])
        
        line1, = ax1.plot([], [], 'o-', lw=LW, markersize=MARKER)
        for i in range(self.n_rod):
            line2, = ax2.plot(self.t, self.y[i*3+2, :], '-', lw=LW, color=colors[i], label='$\\theta_{'+str(i+1)+'}$')
        line2, = ax2.plot([], [], 'o')
        for i, item in enumerate(multipliers):
            line3, = ax3.plot(self.t, item, '-', lw=LW, color=colors[i], label='$\\lambda_{'+str(i+1)+'}$')
        line3, = ax3.plot([], [], 'o')
        
        ax1.set_xlim(( -lim, lim))
        ax1.set_ylim(( -lim, lim))
        ax1.set_aspect('equal', adjustable='box')
        time_template = 'time = {:.2f}s'
        time_text = ax1.text(0.05, 0.9, '', fontsize=FONTSIZE, transform=ax1.transAxes)
        
        ax2.grid()
        ax2.set_ylabel('angle [$rads$]', fontsize=FONTSIZE)
        ax2.set_title('Joint angles', fontsize=FONTSIZE)
        ax3.grid()
        ax3.set_xlabel('Time [s]', fontsize=FONTSIZE)
        ax3.set_ylabel('force [$N$]', fontsize=FONTSIZE)
        ax3.set_title('Lagrangian multipliers', fontsize=FONTSIZE)
        fig.legend(loc=(0.575, 0.4))
        
        for ax in (ax1, ax2, ax3):
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax.xaxis.offsetText.set_fontsize(FONTSIZE)
            ax.yaxis.offsetText.set_fontsize(FONTSIZE)
            
        if title is not None:
            fig.suptitle(title, fontsize=FONTSIZE)
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            return (line1, line2, line3)

        def animate(i):
            x = self.links[:, i, 0]
            y = self.links[:, i, 1]
            line1.set_data(x, y)
            
            line2.set_data([self.t[i]]*self.n_rod, [self.y[n*3+2, i] for n in range(self.n_rod)])
            line2.set_marker(marker='o')
            
            line3.set_data([self.t[i]]*len(multipliers), [multipliers[j, i] for j in range(len(multipliers))]) 
            line3.set_marker(marker='o')
            
            time_text.set_text(time_template.format(self.t[i]))
            return line1, line2, line3, time_text
        
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
            
            if save_as.split('.')[-1] == 'gif':        
                writer = animation.PillowWriter(fps=20)
                anim.save(save_as, writer=writer, dpi=150)
            elif save_as.split('.')[-1] == 'mp4':
                anim.save(save_as, writer='ffmpeg', dpi=150)
            else:
                print('only support gif or mp4')
        
        # return HTML(anim.to_jshtml())
        return HTML(anim.to_html5_video())